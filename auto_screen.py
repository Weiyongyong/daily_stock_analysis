# -*- coding: utf-8 -*-
"""
自动选股模块

参考 DataFetcherManager 的设计模式，实现多数据源自动故障切换：

数据源优先级：
  1. efinance（东方财富爬虫库，API 简洁稳定）
  2. akshare_em（东方财富接口，字段最全）
  3. akshare_sina（新浪财经接口，备用兜底）

每种数据源带独立重试和熔断机制，连续失败后自动跳过，
切换到下一个数据源，确保选股不因单一源故障而中断。

用法:
    python auto_screen.py            # 筛选并打印结果，同时写入 .auto_stock_env
    python auto_screen.py --max 5     # 最多筛选 5 只
    python auto_screen.py --use-history  # 拉历史K线精确计算均线

在 GitHub Actions 中，本脚本在 main.py 之前运行，筛选结果通过
环境变量 STOCK_LIST 直接传递给 main.py（同时写入 .auto_stock_env 供本地使用）。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("auto_screen")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ============================
# 超时配置
# ============================
try:
    _EF_CALL_TIMEOUT = int(os.environ.get("EFINANCE_CALL_TIMEOUT", "30"))
except (ValueError, TypeError):
    _EF_CALL_TIMEOUT = 30

try:
    _AK_CALL_TIMEOUT = int(os.environ.get("AKSHARE_CALL_TIMEOUT", "30"))
except (ValueError, TypeError):
    _AK_CALL_TIMEOUT = 30


# ============================
# 熔断器（参考 realtime_types.CircuitBreaker）
# ============================
class _SimpleCircuitBreaker:
    """
    简化版熔断器，用于自动选股模块。

    状态机：CLOSED → OPEN → HALF_OPEN → CLOSED
    - 连续失败 N 次后进入 OPEN（熔断）
    - 冷却时间后进入 HALF_OPEN（试探）
    - 试探成功 → CLOSED（恢复）
    - 试探失败 → OPEN（继续熔断）
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 300.0,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def _get_state(self, source: str) -> Dict[str, Any]:
        if source not in self._states:
            self._states[source] = {
                "state": self.CLOSED,
                "failures": 0,
                "last_failure_time": 0.0,
            }
        return self._states[source]

    def is_available(self, source: str) -> bool:
        with self._lock:
            state = self._get_state(source)
            now = time.time()

            if state["state"] == self.CLOSED:
                return True

            if state["state"] == self.OPEN:
                if now - state["last_failure_time"] >= self.cooldown_seconds:
                    state["state"] = self.HALF_OPEN
                    logger.info("[熔断器] %s 冷却完成，进入半开状态", source)
                    return True
                return False

            if state["state"] == self.HALF_OPEN:
                return True  # 允许试探

            return True

    def record_success(self, source: str) -> None:
        with self._lock:
            state = self._get_state(source)
            if state["state"] == self.HALF_OPEN:
                logger.info("[熔断器] %s 半开状态成功，恢复正常", source)
            state["state"] = self.CLOSED
            state["failures"] = 0

    def record_failure(self, source: str, error: Optional[str] = None) -> None:
        with self._lock:
            state = self._get_state(source)
            state["failures"] += 1
            state["last_failure_time"] = time.time()

            if state["state"] == self.HALF_OPEN:
                state["state"] = self.OPEN
                logger.warning("[熔断器] %s 半开失败，继续熔断 %.0fs", source, self.cooldown_seconds)
            elif state["failures"] >= self.failure_threshold:
                state["state"] = self.OPEN
                logger.warning("[熔断器] %s 连续失败 %d 次，熔断 %.0fs", source, state["failures"], self.cooldown_seconds)

    def get_status(self) -> Dict[str, str]:
        with self._lock:
            return {s: info["state"] for s, info in self._states.items()}


# 全局熔断器实例
_circuit_breaker = _SimpleCircuitBreaker(
    failure_threshold=3,
    cooldown_seconds=300.0,
)


# ============================
# 带超时的线程调用（参考 efinance_fetcher._ef_call_with_timeout）
# ============================
def _call_with_timeout(func, *args, timeout=None, **kwargs):
    """在带超时的线程中运行库调用，避免无限等待。"""
    if timeout is None:
        timeout = 30
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)


def _compute_ma(df, col: str, window: int) -> "pd.Series":
    """在 DataFrame 上计算指定列的简单移动平均。"""
    return df[col].rolling(window=window, min_periods=1).mean()


# ============================
# 数据源 1: efinance（首选）
# ============================
def _fetch_realtime_efinance(max_retries: int = 3, retry_delay: float = 5.0) -> Optional["pd.DataFrame"]:
    """
    通过 efinance 拉取全 A 股实时行情，带重试机制。
    efinance 使用东方财富数据源，API 封装简洁稳定。
    """
    import efinance as ef

    source = "efinance"
    if not _circuit_breaker.is_available(source):
        logger.info("[熔断] 数据源 %s 处于熔断状态，跳过", source)
        return None

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("正在拉取全 A 股实时行情 (ef.stock.get_realtime_quotes) ... [attempt %d/%d]", attempt, max_retries)
            df = _call_with_timeout(ef.stock.get_realtime_quotes, timeout=_EF_CALL_TIMEOUT)
            if df is not None and not df.empty:
                logger.info("efinance 拉取完成，共 %d 条记录", len(df))
                _circuit_breaker.record_success(source)
                return df
            logger.warning("efinance 返回空数据 [attempt %d/%d]", attempt, max_retries)
        except FuturesTimeoutError:
            last_error = TimeoutError(f"efinance 调用超时 (>{_EF_CALL_TIMEOUT}s)")
            logger.warning("efinance 拉取超时 [attempt %d/%d]: %s", attempt, max_retries, last_error)
        except Exception as e:
            last_error = e
            logger.warning("efinance 拉取失败 [attempt %d/%d]: %s", attempt, max_retries, e)

        if attempt < max_retries:
            wait = retry_delay * attempt
            logger.info("等待 %.0f 秒后重试...", wait)
            time.sleep(wait)

    _circuit_breaker.record_failure(source, str(last_error))
    logger.error("efinance 拉取最终失败（共重试 %d 次）: %s", max_retries, last_error)
    return None


# ============================
# 数据源 2: akshare 东方财富接口（次选）
# ============================
def _fetch_realtime_akshare_em(max_retries: int = 3, retry_delay: float = 5.0) -> Optional["pd.DataFrame"]:
    """
    通过 akshare 东方财富接口拉取全 A 股实时行情，带重试机制。
    字段最全（含 60日涨跌幅、昨收 等），但海外服务器可能连接不稳定。
    """
    import akshare as ak

    source = "akshare_em"
    if not _circuit_breaker.is_available(source):
        logger.info("[熔断] 数据源 %s 处于熔断状态，跳过", source)
        return None

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("正在拉取全 A 股实时行情 (ak.stock_zh_a_spot_em) ... [attempt %d/%d]", attempt, max_retries)
            df = _call_with_timeout(ak.stock_zh_a_spot_em, timeout=_AK_CALL_TIMEOUT)
            if df is not None and not df.empty:
                logger.info("akshare_em 拉取完成，共 %d 条记录", len(df))
                _circuit_breaker.record_success(source)
                return df
            logger.warning("akshare_em 返回空数据 [attempt %d/%d]", attempt, max_retries)
        except FuturesTimeoutError:
            last_error = TimeoutError(f"akshare_em 调用超时 (>{_AK_CALL_TIMEOUT}s)")
            logger.warning("akshare_em 拉取超时 [attempt %d/%d]", attempt, max_retries)
        except Exception as e:
            last_error = e
            logger.warning("akshare_em 拉取失败 [attempt %d/%d]: %s", attempt, max_retries, e)

        if attempt < max_retries:
            wait = retry_delay * attempt
            logger.info("等待 %.0f 秒后重试...", wait)
            time.sleep(wait)

    _circuit_breaker.record_failure(source, str(last_error))
    logger.error("akshare_em 拉取最终失败: %s", last_error)
    return None


# ============================
# 数据源 3: akshare 新浪接口（兜底）
# ============================
def _fetch_realtime_akshare_sina(max_retries: int = 3, retry_delay: float = 5.0) -> Optional["pd.DataFrame"]:
    """
    通过 akshare 新浪接口拉取全 A 股实时行情，带重试机制。
    新浪接口字段较少，但连接通常更稳定，作为最后兜底。
    """
    import akshare as ak

    source = "akshare_sina"
    if not _circuit_breaker.is_available(source):
        logger.info("[熔断] 数据源 %s 处于熔断状态，跳过", source)
        return None

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("正在拉取全 A 股实时行情 (ak.stock_zh_a_spot 新浪) ... [attempt %d/%d]", attempt, max_retries)
            df = _call_with_timeout(ak.stock_zh_a_spot, timeout=_AK_CALL_TIMEOUT)
            if df is not None and not df.empty:
                logger.info("akshare_sina 拉取完成，共 %d 条记录", len(df))
                _circuit_breaker.record_success(source)
                return df
            logger.warning("akshare_sina 返回空数据 [attempt %d/%d]", attempt, max_retries)
        except FuturesTimeoutError:
            last_error = TimeoutError(f"akshare_sina 调用超时 (>{_AK_CALL_TIMEOUT}s)")
            logger.warning("akshare_sina 拉取超时 [attempt %d/%d]", attempt, max_retries)
        except Exception as e:
            last_error = e
            logger.warning("akshare_sina 拉取失败 [attempt %d/%d]: %s", attempt, max_retries, e)

        if attempt < max_retries:
            wait = retry_delay * attempt
            logger.info("等待 %.0f 秒后重试...", wait)
            time.sleep(wait)

    _circuit_breaker.record_failure(source, str(last_error))
    logger.error("akshare_sina 拉取最终失败: %s", last_error)
    return None


# ============================
# 多数据源自动切换（参考 DataFetcherManager 故障切换策略）
# ============================
def _fetch_realtime_with_fallback() -> Optional["pd.DataFrame"]:
    """
    依次尝试多个数据源拉取全 A 股行情，自动故障切换。

    优先级：efinance → akshare_em → akshare_sina
    每个数据源有独立熔断器，连续失败后自动跳过。
    """
    sources = [
        ("efinance", _fetch_realtime_efinance),
        ("akshare_em", _fetch_realtime_akshare_em),
        ("akshare_sina", _fetch_realtime_akshare_sina),
    ]

    failed_sources: List[str] = []
    for source_name, fetch_func in sources:
        if not _circuit_breaker.is_available(source_name):
            logger.info("[选股] 数据源 %s 熔断中，跳过", source_name)
            failed_sources.append(source_name)
            continue

        logger.info("[选股] 尝试数据源: %s", source_name)
        df = fetch_func()
        if df is not None and not df.empty:
            if failed_sources:
                logger.info("[选股] 首选数据源 %s 失败，从 %s fallback 成功", failed_sources[0], source_name)
            return df

        failed_sources.append(source_name)
        logger.info("[选股] 数据源 %s 不可用，尝试下一个...", source_name)

    logger.error("[选股] 所有数据源均不可用: %s", ", ".join(failed_sources))
    logger.info("[选股] 熔断器状态: %s", _circuit_breaker.get_status())
    return None


# ============================
# 列名统一（参考 DataFetcherManager 的 _normalize_columns 模式）
# ============================
# 各数据源返回的列名差异：
# efinance:     股票代码, 股票名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 开盘, 换手率, 量比, 市盈率, 总市值, 流通市值, 昨收
# akshare_em:   代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 开盘, 换手率, 量比, 市盈率, 总市值, 流通市值, 60日涨跌幅, 昨收
# akshare_sina: code, name, trade, changepercent, tradevolume, amount, turnoverratio, ...

_UNIFIED_COLUMNS = {
    # 代码与名称
    "股票代码": "代码", "code": "代码",
    "股票名称": "名称", "name": "名称",
    # 价格
    "最新价": "最新价", "trade": "最新价", "close": "最新价",
    # 涨跌
    "涨跌幅": "涨跌幅", "changepercent": "涨跌幅", "pct_chg": "涨跌幅",
    "涨跌额": "涨跌额", "change": "涨跌额", "changeamount": "涨跌额",
    # 量
    "成交量": "成交量", "tradevolume": "成交量", "volume": "成交量", "vol": "成交量",
    "成交额": "成交额", "amount": "成交额",
    # 比率
    "振幅": "振幅", "amplitude": "振幅",
    "换手率": "换手率", "turnoverratio": "换手率", "turnover_rate": "换手率",
    "量比": "量比", "volume_ratio": "量比",
    # 价格区间
    "最高": "最高", "high": "最高",
    "最低": "最低", "low": "最低",
    "开盘": "开盘", "open": "开盘", "今开": "开盘",
    "昨收": "昨收", "pre_close": "昨收", "lastclose": "昨收",
    # 估值
    "市盈率": "市盈率", "pe_ratio": "市盈率",
    "总市值": "总市值", "total_mv": "总市值",
    "流通市值": "流通市值", "circ_mv": "流通市值",
    # 特殊字段（仅 akshare_em 有）
    "60日涨跌幅": "60日涨跌幅",
}


def _normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    统一不同数据源返回的列名为内部统一名称。

    参考 DataFetcherManager 的 _normalize_columns 模式：
    各数据源列名差异大，统一为内部名称后才能在同一筛选逻辑中使用。
    """
    rename = {}
    for raw_col, unified_col in _UNIFIED_COLUMNS.items():
        if raw_col in df.columns and unified_col not in df.columns:
            # 避免冲突：如果同一 unified_col 有多个 raw_col 映射，优先已有的
            if unified_col not in rename.values():
                rename[raw_col] = unified_col

    if rename:
        df = df.rename(columns=rename)
        logger.debug("[列名统一] 重命名: %s", rename)

    # 确保核心列存在（缺失时填充默认值，避免后续筛选报错）
    essential_cols = ["代码", "名称", "涨跌幅", "成交量", "成交额"]
    for col in essential_cols:
        if col not in df.columns:
            logger.warning("[列名统一] 数据缺少核心列 '%s'", col)

    return df


# ============================
# 历史K线拉取（支持 efinance + akshare 双源）
# ============================
def _fetch_history_efinance(code: str, days: int = 30) -> Optional["pd.DataFrame"]:
    """通过 efinance 拉取历史日K线。"""
    import efinance as ef
    from datetime import datetime, timedelta

    beg = (datetime.now() - timedelta(days=days + 20)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    try:
        df = _call_with_timeout(
            ef.stock.get_quote_history,
            stock_codes=code,
            beg=beg,
            end=end,
            klt=101,  # 日线
            fqt=1,    # 前复权
            timeout=60,
        )
        if df is not None and not df.empty and "收盘" in df.columns:
            return df
        return None
    except Exception as e:
        logger.debug("efinance 拉取 %s 历史K线失败: %s", code, e)
        return None


def _fetch_history_akshare(code: str, days: int = 30) -> Optional["pd.DataFrame"]:
    """通过 akshare 拉取历史日K线（兜底）。"""
    import akshare as ak
    from datetime import datetime, timedelta

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 20)).strftime("%Y%m%d")
    try:
        df = _call_with_timeout(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
            timeout=60,
        )
        if df is not None and not df.empty and "收盘" in df.columns:
            return df
        return None
    except Exception as e:
        logger.debug("akshare 拉取 %s 历史K线失败: %s", code, e)
        return None


def _fetch_history(code: str, days: int = 30) -> Optional["pd.DataFrame"]:
    """
    拉取历史K线，优先 efinance，失败后 akshare 兜底。

    参考 DataFetcherManager 的 get_daily_data 故障切换模式。
    """
    # 优先 efinance
    df = _fetch_history_efinance(code, days=days)
    if df is not None:
        return df

    # akshare 兜底
    df = _fetch_history_akshare(code, days=days)
    if df is not None:
        logger.info("[历史K线] %s 从 akshare fallback 成功", code)
        return df

    logger.debug("[历史K线] %s 所有数据源均不可用", code)
    return None


# ============================
# 筛选策略
# ============================
def _screen_with_history(df_spot: "pd.DataFrame", max_count: int = 10) -> List[str]:
    """
    策略一（精确版）：逐只拉取历史 K 线计算均线。
    先用实时快照做粗筛，再对粗筛结果逐一拉历史 K 线确认均线多头排列。
    """
    import pandas as pd

    df = _normalize_columns(df_spot.copy())

    # 过滤 ST、退市
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", regex=True, na=False)]
    # 涨跌幅有效
    if "涨跌幅" not in df.columns:
        logger.warning("缺少涨跌幅列，无法粗筛")
        return []

    df = df[df["涨跌幅"] != "-"]
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df.dropna(subset=["涨跌幅"])

    # 粗筛条件：涨跌幅温和（不追高）
    rough = df[(df["涨跌幅"] >= -2) & (df["涨跌幅"] <= 5)]
    # 按成交额降序，优先关注活跃股
    if "成交额" in rough.columns:
        rough["成交额"] = pd.to_numeric(rough["成交额"], errors="coerce")
        rough = rough.sort_values("成交额", ascending=False)

    # 最多取前 50 只做精细筛选
    candidates = rough.head(50)
    logger.info("粗筛后候选股数量: %d", len(candidates))

    # --- 精筛：逐只拉历史 K 线计算 MA ---
    results: List[str] = []
    for _, row in candidates.iterrows():
        code = str(row["代码"])
        if len(results) >= max_count:
            break
        try:
            hist = _fetch_history(code, days=30)
            if hist is None or hist.empty:
                continue
            hist["MA5"] = _compute_ma(hist, "收盘", 5)
            hist["MA10"] = _compute_ma(hist, "收盘", 10)
            hist["MA20"] = _compute_ma(hist, "收盘", 20)
            hist["MA5_volume"] = _compute_ma(hist, "成交量", 5)

            last = hist.iloc[-1]
            # 均线多头排列
            if not (last["MA5"] > last["MA10"] > last["MA20"]):
                continue
            # 放量：当日成交量 > 5 日均量
            if last["成交量"] <= last["MA5_volume"]:
                continue

            results.append(code)
            logger.info("入选: %s %s  价格=%.2f  涨跌幅=%.2f%%", code, row.get("名称", ""), last["收盘"], row["涨跌幅"])
        except Exception as e:
            logger.debug("跳过 %s: %s", code, e)
            continue

    return results


def _screen_quick(df_spot: "pd.DataFrame", max_count: int = 10) -> List[str]:
    """
    策略二（快速版）：仅用实时快照中可用字段做筛选，不拉历史 K 线。

    利用"量比 > 0.8"（efinance/akshare_em）或"60日涨跌幅 > 0"（仅 akshare_em）
    作为趋势向上粗判，适合 GitHub Actions 时间受限场景。
    """
    import pandas as pd

    df = _normalize_columns(df_spot.copy())

    # 过滤 ST、退市
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", regex=True, na=False)]

    # 确保涨跌幅列存在
    if "涨跌幅" not in df.columns:
        logger.error("数据缺少涨跌幅列，无法筛选")
        return []

    df = df[df["涨跌幅"] != "-"]
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df.dropna(subset=["涨跌幅"])

    for col in ("成交量", "成交额", "换手率", "量比", "60日涨跌幅"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 构建筛选条件
    cond = (
        # 涨跌幅温和，不追高
        (df["涨跌幅"] >= -2)
        & (df["涨跌幅"] <= 5)
    )

    # 成交量 > 0（排除停牌）
    if "成交量" in df.columns:
        cond = cond & (df["成交量"] > 0)

    # 量比 >= 0.8（有一定资金关注度）—— efinance 和 akshare_em 都有此字段
    if "量比" in df.columns:
        cond = cond & (df["量比"] >= 0.8)

    # 60日涨跌幅为正（中期趋势向上）—— 仅 akshare_em 有此字段
    if "60日涨跌幅" in df.columns:
        cond = cond & (df["60日涨跌幅"] > 0)

    # 换手率在合理区间（1%~15%）
    if "换手率" in df.columns:
        cond = cond & (df["换手率"] >= 1) & (df["换手率"] <= 15)

    screen_result = df[cond]

    # 按成交额降序
    if "成交额" in screen_result.columns:
        screen_result = screen_result.sort_values("成交额", ascending=False)

    stock_code_list = screen_result["代码"].head(max_count).tolist()
    return stock_code_list


# ============================
# 主入口
# ============================
def auto_screen_stocks(
    max_count: int = 10,
    use_history: bool = False,
) -> str:
    """
    执行自动选股，返回逗号分隔的股票代码字符串。

    Args:
        max_count: 最多返回的股票数量
        use_history: 是否拉取历史K线计算均线（精确但慢）

    Returns:
        逗号分隔的股票代码字符串，如 "600519,000001"；
        选股失败时返回空字符串，由调用方决定是否回退到默认配置。
    """
    df = _fetch_realtime_with_fallback()

    if df is None:
        logger.error("无法获取行情数据，自动选股失败，将回退到默认 STOCK_LIST 配置")
        logger.info("熔断器状态: %s", _circuit_breaker.get_status())
        return ""

    # 记录实际使用的数据源（通过列名特征判断）
    source_detected = _detect_source(df)
    logger.info("实际使用的数据源: %s", source_detected)

    try:
        if use_history:
            stock_code_list = _screen_with_history(df, max_count=max_count)
        else:
            stock_code_list = _screen_quick(df, max_count=max_count)
    except Exception as e:
        logger.error("选股筛选过程出错: %s", e)
        return ""

    if not stock_code_list:
        logger.warning("未筛选到符合条件的股票，将使用默认股票池")
        return ""

    stock_str = ",".join(stock_code_list)
    logger.info("今日自动筛选股票池（%d 只）: %s", len(stock_code_list), stock_str)

    # 写入临时文件（供本地运行 main.py 时读取）
    env_path = os.path.join(os.getcwd(), ".auto_stock_env")
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"STOCK_LIST={stock_str}")
        logger.info("选股结果已写入 %s", env_path)
    except OSError as e:
        logger.warning("写入 .auto_stock_env 失败: %s", e)

    # GitHub Actions / 本地：直接设置环境变量，供同进程后续读取
    os.environ["STOCK_LIST"] = stock_str

    return stock_str


def _detect_source(df: "pd.DataFrame") -> str:
    """通过列名特征判断数据来自哪个源（辅助调试）。"""
    cols = set(df.columns)
    if "股票代码" in cols and "量比" in cols and "昨收" in cols and "60日涨跌幅" not in cols:
        return "efinance"
    if "代码" in cols and "60日涨跌幅" in cols:
        return "akshare_em"
    if "code" in cols and "trade" in cols and "changepercent" in cols:
        return "akshare_sina"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="A股自动选股（多数据源自动切换）")
    parser.add_argument("--max", type=int, default=10, help="最多筛选的股票数量（默认 10）")
    parser.add_argument(
        "--use-history",
        action="store_true",
        help="拉取历史K线精确计算均线（较慢，默认关闭）",
    )
    args = parser.parse_args()

    stock_str = auto_screen_stocks(max_count=args.max, use_history=args.use_history)
    if stock_str:
        print(f"STOCK_LIST={stock_str}")
        return 0
    else:
        print("未筛选到符合条件的股票或行情数据获取失败，将使用默认 STOCK_LIST 配置")
        # 返回 0 而非 1，使 GitHub Actions 不会因选股失败而中断后续步骤
        return 0


if __name__ == "__main__":
    sys.exit(main())
