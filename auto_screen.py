# -*- coding: utf-8 -*-
"""
自动选股模块

支持两种选股引擎，通过环境变量 SCREEN_METHOD 切换：

  1. auto_screen（默认）：内置多数据源 + 技术指标筛选
     数据源优先级：
       a. tencent_batch（腾讯批量实时接口，海外稳定、秒级响应，首选）
       b. efinance（东方财富爬虫库，API 简洁稳定）
       c. akshare_em（东方财富接口，字段最全）
       d. akshare_sina（新浪财经接口，备用兜底，超时 120s）
     筛选策略：涨跌幅温和 + 量比/换手率/成交额 + 均线多头排列（可选）

  2. alphasift：AlphaSift 多因子 + LLM 排序选股引擎
     通过 src.services.alphasift_service.AlphaSiftService.screen() 调用
     需要 ALPHASIFT_ENABLED=true + LLM 配置

  3. alphasift_fallback：先 AlphaSift，失败 fallback 到 auto_screen
  4. auto_screen_fallback：先 auto_screen，失败 fallback 到 AlphaSift

用法:
    python auto_screen.py                         # 默认 auto_screen 选股
    python auto_screen.py --max 5                  # 最多 5 只
    python auto_screen.py --use-history             # 拉历史K线算均线
    python auto_screen.py --method alphasift        # 使用 AlphaSift 选股
    python auto_screen.py --method alphasift_fallback  # AlphaSift 失败则 fallback

环境变量:
    SCREEN_METHOD          选股引擎（auto_screen/alphasift/alphasift_fallback/auto_screen_fallback）
    ALPHASIFT_ENABLED      是否启用 AlphaSift（true/false）
    ALPHASIFT_STRATEGY     AlphaSift 策略（默认 dual_low）
    ALPHASIFT_MARKET       AlphaSift 市场（默认 cn）

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

# akshare_sina 分页拉取慢，超时放宽到 120s
try:
    _AK_SINA_TIMEOUT = int(os.environ.get("AKSHARE_SINA_TIMEOUT", "120"))
except (ValueError, TypeError):
    _AK_SINA_TIMEOUT = 120

# 腾讯批量接口超时（每次请求约 800 只，通常 <10s）
try:
    _TENCENT_BATCH_TIMEOUT = int(os.environ.get("TENCENT_BATCH_TIMEOUT", "30"))
except (ValueError, TypeError):
    _TENCENT_BATCH_TIMEOUT = 30


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
# 数据源 0: 腾讯批量实时接口（首选，海外稳定）
# ============================
# 腾讯接口 http://qt.gtimg.cn/q=sh600519,sz000001,... 支持批量查询
# 每次请求最多约 900 只股票，海外连接稳定、响应快（秒级）
# 参考 data_provider/akshare_fetcher.py 的 _get_stock_realtime_quote_tencent

_TENCENT_ENDPOINT = "http://qt.gtimg.cn/q="
_TENCENT_BATCH_SIZE = 800  # 每批最多查询数量


def _to_tencent_symbol(code: str) -> str:
    """将 6 位 A 股代码转换为腾讯格式 sh600519 / sz000001 / bj920748。"""
    base = code.strip().split(".")[0] if "." in code else code.strip()
    if base.startswith(("8", "4", "9")):  # 北交所
        return f"bj{base}"
    if base.startswith(("6", "5", "9")):
        return f"sh{base}"
    return f"sz{base}"


def _parse_tencent_batch_response(text: str) -> List[Dict[str, Any]]:
    """
    解析腾讯批量行情响应文本，返回统一列名字典列表。

    腾讯字段顺序（~分隔）：
      1:名称 2:代码 3:最新价 4:昨收 5:今开 6:成交量 7:外盘 8:内盘
      9-28:买卖五档 30:时间戳 31:涨跌额 32:涨跌幅(%) 33:最高 34:最低
      35:收盘/成交量/成交额 36:成交量(口径随 payload 变化) 37:成交额(万)
      38:换手率(%) 39:市盈率 43:振幅(%) 44:流通市值(亿) 45:总市值(亿)
      46:市净率 47:涨停价 48:跌停价 49:量比
    """
    import re

    results: List[Dict[str, Any]] = []
    # 腾讯响应格式：v_sh600519="1~贵州茅台~600519~1680.00~...~";
    blocks = text.split(";")
    for block in blocks:
        block = block.strip()
        if not block or '=""' in block:
            continue
        # 提取引号内数据
        match = re.search(r'"([^"]+)"', block)
        if not match:
            continue
        data_str = match.group(1)
        fields = data_str.split("~")
        if len(fields) < 35:
            continue

        def _safe_float(idx: int) -> Optional[float]:
            if idx >= len(fields) or not fields[idx]:
                return None
            try:
                return float(fields[idx])
            except (ValueError, TypeError):
                return None

        def _safe_int(idx: int) -> Optional[int]:
            val = _safe_float(idx)
            return int(val) if val is not None else None

        row = {
            "代码": fields[2] if len(fields) > 2 else "",
            "名称": fields[1] if len(fields) > 1 else "",
            "最新价": _safe_float(3),
            "昨收": _safe_float(4),
            "开盘": _safe_float(5),
            "成交量": _safe_int(6),  # 成交量（手）
            "涨跌额": _safe_float(31),
            "涨跌幅": _safe_float(32),
            "最高": _safe_float(33),
            "最低": _safe_float(34),
            "成交额": _safe_float(37),  # 成交额（万元）→ 后续转换为元
            "换手率": _safe_float(38),
            "市盈率": _safe_float(39),
            "振幅": _safe_float(43),
            "流通市值": _safe_float(44),  # 亿
            "总市值": _safe_float(45),  # 亿
            "量比": _safe_float(49),
        }
        results.append(row)
    return results


def _generate_all_a_share_codes() -> List[str]:
    """
    生成全 A 股 6 位代码列表。

    范围：
      - 沪市主板: 600000-609999
      - 沪市科创板: 688000-689999
      - 深市主板: 000001-003999
      - 深市创业板: 300000-301999
      - 北交所: 830000-920999
    """
    codes: List[str] = []

    # 沪市主板 600000-609999
    for i in range(600000, 610000):
        codes.append(str(i))
    # 科创板 688000-689999
    for i in range(688000, 690000):
        codes.append(str(i))
    # 深市主板 000001-003999
    for i in range(1, 4000):
        codes.append(f"{i:06d}")
    # 创业板 300000-301999
    for i in range(300000, 302000):
        codes.append(str(i))
    # 北交所 830000-839999, 870000-879999, 920000-920999
    for i in range(830000, 840000):
        codes.append(str(i))
    for i in range(870000, 880000):
        codes.append(str(i))
    for i in range(920000, 921000):
        codes.append(str(i))

    return codes


def _fetch_realtime_tencent_batch(max_retries: int = 3, retry_delay: float = 3.0) -> Optional["pd.DataFrame"]:
    """
    通过腾讯批量实时接口拉取全 A 股行情。

    腾讯接口 http://qt.gtimg.cn/q=sh600519,sz000001,... 支持批量查询，
    海外服务器连接稳定、响应快（每批 <10s），适合 GitHub Actions 环境。

    策略：生成所有可能的 A 股代码，分批请求腾讯接口，过滤无效响应。
    """
    import pandas as pd
    import requests

    source = "tencent_batch"
    if not _circuit_breaker.is_available(source):
        logger.info("[熔断] 数据源 %s 处于熔断状态，跳过", source)
        return None

    all_codes = _generate_all_a_share_codes()
    logger.info(
        "腾讯批量接口：共 %d 个候选代码，分批查询（每批 %d 只）",
        len(all_codes),
        _TENCENT_BATCH_SIZE,
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://finance.qq.com",
    }

    last_error: Optional[Exception] = None
    all_rows: List[Dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        try:
            all_rows = []
            total_batches = (len(all_codes) + _TENCENT_BATCH_SIZE - 1) // _TENCENT_BATCH_SIZE

            for batch_idx in range(total_batches):
                start = batch_idx * _TENCENT_BATCH_SIZE
                end = min(start + _TENCENT_BATCH_SIZE, len(all_codes))
                batch_codes = all_codes[start:end]

                # 转换为腾讯格式
                symbols = [_to_tencent_symbol(c) for c in batch_codes]
                query = ",".join(symbols)
                url = f"{_TENCENT_ENDPOINT}{query}"

                logger.info(
                    "腾讯批量请求 [%d/%d] 代码 %d-%d ...",
                    batch_idx + 1,
                    total_batches,
                    start,
                    end,
                )

                resp = requests.get(url, headers=headers, timeout=_TENCENT_BATCH_TIMEOUT)
                resp.encoding = "gbk"

                rows = _parse_tencent_batch_response(resp.text)
                # 过滤无效数据（价格为 0 或 None 说明代码不存在）
                valid_rows = [r for r in rows if r.get("最新价") and r["最新价"] > 0]
                all_rows.extend(valid_rows)

                logger.info(
                    "  批次 %d: 解析 %d 条，有效 %d 条",
                    batch_idx + 1,
                    len(rows),
                    len(valid_rows),
                )

            if all_rows:
                df = pd.DataFrame(all_rows)
                # 成交额从万元转换为元
                if "成交额" in df.columns:
                    df["成交额"] = df["成交额"] * 10000
                # 市值从亿转换为元
                if "流通市值" in df.columns:
                    df["流通市值"] = df["流通市值"] * 100000000
                if "总市值" in df.columns:
                    df["总市值"] = df["总市值"] * 100000000

                logger.info("腾讯批量拉取完成，共 %d 只有效股票", len(df))
                _circuit_breaker.record_success(source)
                return df

            logger.warning("腾讯批量拉取返回空数据 [attempt %d/%d]", attempt, max_retries)

        except Exception as e:
            last_error = e
            logger.warning("腾讯批量拉取失败 [attempt %d/%d]: %s", attempt, max_retries, e)

        if attempt < max_retries:
            wait = retry_delay * attempt
            logger.info("等待 %.0f 秒后重试...", wait)
            time.sleep(wait)

    _circuit_breaker.record_failure(source, str(last_error))
    logger.error("腾讯批量拉取最终失败: %s", last_error)
    return None


# ============================
# 数据源 1: efinance（次选）
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
            df = _call_with_timeout(ak.stock_zh_a_spot, timeout=_AK_SINA_TIMEOUT)
            if df is not None and not df.empty:
                logger.info("akshare_sina 拉取完成，共 %d 条记录", len(df))
                _circuit_breaker.record_success(source)
                return df
            logger.warning("akshare_sina 返回空数据 [attempt %d/%d]", attempt, max_retries)
        except FuturesTimeoutError:
            last_error = TimeoutError(f"akshare_sina 调用超时 (>{_AK_SINA_TIMEOUT}s)")
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

    优先级：tencent_batch → efinance → akshare_em → akshare_sina
    每个数据源有独立熔断器，连续失败后自动跳过。
    """
    sources = [
        ("tencent_batch", _fetch_realtime_tencent_batch),
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
# tencent_batch: 代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 开盘, 换手率, 量比, 市盈率, 总市值, 流通市值, 昨收（本模块自行解析，已统一）
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
# AlphaSift 选股引擎
# ============================
def _screen_alphasift(max_count: int = 10) -> List[str]:
    """
    通过 AlphaSift 选股引擎进行选股。

    AlphaSift 是一个基于多因子 + LLM 排序的选股框架，
    通过 src.services.alphasift_service.AlphaSiftService.screen() 调用。

    需要:
      - ALPHASIFT_ENABLED=true
      - alphasift 包已安装（requirements.txt 中已包含）
      - LLM 配置（用于 LLM 排序，可选但推荐）

    Args:
        max_count: 最多返回的股票数量

    Returns:
        股票代码列表，失败时返回空列表
    """
    try:
        from src.config import Config
        from src.services.alphasift_service import AlphaSiftService
    except ImportError as e:
        logger.error("[AlphaSift] 导入 AlphaSift 服务失败: %s", e)
        return []

    # 确保 ALPHASIFT_ENABLED=true
    if not os.environ.get("ALPHASIFT_ENABLED", "").lower() in ("true", "1", "yes"):
        logger.warning("[AlphaSift] ALPHASIFT_ENABLED 未开启，跳过 AlphaSift 选股")
        return []

    try:
        config = Config.get_instance()
    except Exception as e:
        logger.error("[AlphaSift] 加载配置失败: %s", e)
        return []

    if not config.alphasift_enabled:
        logger.warning("[AlphaSift] config.alphasift_enabled=False，跳过 AlphaSift 选股")
        return []

    strategy = os.environ.get("ALPHASIFT_STRATEGY", "dual_low")
    market = os.environ.get("ALPHASIFT_MARKET", "cn")

    logger.info(
        "[AlphaSift] 开始选股: strategy=%s, market=%s, max_results=%d",
        strategy, market, max_count,
    )

    try:
        service = AlphaSiftService(config=config)
        result = service.screen(
            strategy=strategy,
            market=market,
            max_results=max_count,
        )
    except Exception as e:
        logger.error("[AlphaSift] 选股调用失败: %s", e)
        return []

    candidates = result.get("candidates") or []
    if not candidates:
        logger.warning("[AlphaSift] 选股结果为空 (candidate_count=%s)", result.get("candidate_count"))
        return []

    stock_codes: List[str] = []
    for candidate in candidates:
        code = str(candidate.get("code") or "").strip()
        if code:
            # 标准化为 6 位纯数字代码
            clean_code = code.split(".")[0].replace("sh", "").replace("sz", "").replace("bj", "")
            if clean_code.isdigit() and len(clean_code) == 6:
                stock_codes.append(clean_code)
                name = candidate.get("name", "")
                score = candidate.get("score", "")
                logger.info(
                    "[AlphaSift] 入选: %s %s  score=%s  rank=%s",
                    clean_code, name, score, candidate.get("rank", ""),
                )
            if len(stock_codes) >= max_count:
                break

    logger.info("[AlphaSift] 选股完成，共 %d 只: %s", len(stock_codes), ",".join(stock_codes))
    return stock_codes


# ============================
# 主入口
# ============================
def auto_screen_stocks(
    max_count: int = 10,
    use_history: bool = False,
) -> str:
    """
    执行自动选股，返回逗号分隔的股票代码字符串。

    支持通过环境变量 SCREEN_METHOD 切换选股引擎：
      - auto_screen（默认）：使用内置多数据源筛选策略
      - alphasift：使用 AlphaSift 选股引擎（需 ALPHASIFT_ENABLED=true）
      - alphasift_fallback：先 AlphaSift，失败则 fallback 到 auto_screen
      - auto_screen_fallback：先 auto_screen，失败则 fallback 到 AlphaSift

    Args:
        max_count: 最多返回的股票数量
        use_history: 是否拉取历史K线计算均线（精确但慢，仅 auto_screen 模式生效）

    Returns:
        逗号分隔的股票代码字符串，如 "600519,000001"；
        选股失败时返回空字符串，由调用方决定是否回退到默认配置。
    """
    screen_method = os.environ.get("SCREEN_METHOD", "auto_screen").strip().lower()
    logger.info("[选股] 选股引擎: %s", screen_method)

    stock_code_list: List[str] = []

    if screen_method == "alphasift":
        stock_code_list = _screen_alphasift(max_count=max_count)

    elif screen_method == "alphasift_fallback":
        stock_code_list = _screen_alphasift(max_count=max_count)
        if not stock_code_list:
            logger.warning("[选股] AlphaSift 选股失败，fallback 到 auto_screen")
            stock_code_list = _auto_screen_internal(max_count, use_history)

    elif screen_method == "auto_screen_fallback":
        stock_code_list = _auto_screen_internal(max_count, use_history)
        if not stock_code_list:
            logger.warning("[选股] auto_screen 选股失败，fallback 到 AlphaSift")
            stock_code_list = _screen_alphasift(max_count=max_count)

    else:  # auto_screen (默认)
        stock_code_list = _auto_screen_internal(max_count, use_history)

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


def _auto_screen_internal(max_count: int, use_history: bool) -> List[str]:
    """
    内置自动选股逻辑（多数据源 + 技术指标筛选）。

    从 _fetch_realtime_with_fallback 拉取行情，然后根据 use_history
    选择精确版（拉历史K线算均线）或快速版（仅用实时快照）筛选。
    """
    df = _fetch_realtime_with_fallback()

    if df is None:
        logger.error("无法获取行情数据，自动选股失败，将回退到默认 STOCK_LIST 配置")
        logger.info("熔断器状态: %s", _circuit_breaker.get_status())
        return []

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
        return []

    return stock_code_list


def _detect_source(df: "pd.DataFrame") -> str:
    """通过列名特征判断数据来自哪个源（辅助调试）。"""
    cols = set(df.columns)
    # 腾讯批量源的特征：列名为中文但来自我们自己解析，含"昨收"但不含"60日涨跌幅"
    # 且不含"股票代码"（efinance 用"股票代码"而非"代码"）
    if "代码" in cols and "昨收" in cols and "量比" in cols and "60日涨跌幅" not in cols and "股票代码" not in cols:
        return "tencent_batch"
    if "股票代码" in cols and "量比" in cols and "昨收" in cols and "60日涨跌幅" not in cols:
        return "efinance"
    if "代码" in cols and "60日涨跌幅" in cols:
        return "akshare_em"
    if "code" in cols and "trade" in cols and "changepercent" in cols:
        return "akshare_sina"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="A股自动选股（支持 auto_screen / AlphaSift 切换）")
    parser.add_argument("--max", type=int, default=10, help="最多筛选的股票数量（默认 10）")
    parser.add_argument(
        "--use-history",
        action="store_true",
        help="拉取历史K线精确计算均线（较慢，默认关闭，仅 auto_screen 模式生效）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="",
        choices=["", "auto_screen", "alphasift", "alphasift_fallback", "auto_screen_fallback"],
        help="选股引擎: auto_screen（默认）| alphasift | alphasift_fallback | auto_screen_fallback。"
             "留空则读取环境变量 SCREEN_METHOD",
    )
    args = parser.parse_args()

    # 命令行 --method 优先于环境变量 SCREEN_METHOD
    if args.method:
        os.environ["SCREEN_METHOD"] = args.method

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
