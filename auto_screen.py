# -*- coding: utf-8 -*-
"""
自动选股模块

通过 efinance 拉取全 A 股实时行情，并结合历史 K 线计算均线，
按"均线多头排列 + 放量上涨 + 涨跌幅温和"策略筛选候选股。

数据源：efinance（东方财富爬虫库，https://github.com/Micro-sheep/efinance）
  - ef.stock.get_realtime_quotes()  全 A 股实时行情
  - ef.stock.get_quote_history()    历史日K线

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
from typing import List, Optional

logger = logging.getLogger("auto_screen")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# efinance 调用超时（秒），防止海外服务器访问东方财富时无限等待
try:
    _EF_CALL_TIMEOUT = int(os.environ.get("EFINANCE_CALL_TIMEOUT", "30"))
except (ValueError, TypeError):
    _EF_CALL_TIMEOUT = 30


def _ef_call_with_timeout(func, *args, timeout=None, **kwargs):
    """在带超时的线程中运行 efinance 调用，避免无限等待。"""
    if timeout is None:
        timeout = _EF_CALL_TIMEOUT
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)


def _compute_ma(df, col: str, window: int) -> "pd.Series":
    """在 DataFrame 上计算指定列的简单移动平均。"""
    return df[col].rolling(window=window, min_periods=1).mean()


def _fetch_realtime(max_retries: int = 3, retry_delay: float = 5.0) -> Optional["pd.DataFrame"]:
    """
    通过 efinance 拉取全 A 股实时快照行情，带重试机制。

    efinance 使用东方财富数据源，与 akshare 的 stock_zh_a_spot_em 同源，
    但 efinance 的请求封装更稳定，且支持超时控制。
    """
    import efinance as ef

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("正在拉取全 A 股实时行情 (ef.stock.get_realtime_quotes) ... [attempt %d/%d]", attempt, max_retries)
            df = _ef_call_with_timeout(ef.stock.get_realtime_quotes)
            if df is not None and not df.empty:
                logger.info("拉取完成，共 %d 条记录", len(df))
                return df
            logger.warning("ef.stock.get_realtime_quotes 返回空数据 [attempt %d/%d]", attempt, max_retries)
        except FuturesTimeoutError:
            last_error = TimeoutError(f"efinance 调用超时 (>{_EF_CALL_TIMEOUT}s)")
            logger.warning("拉取超时 [attempt %d/%d]: %s", attempt, max_retries, last_error)
        except Exception as e:
            last_error = e
            logger.warning("拉取失败 [attempt %d/%d]: %s", attempt, max_retries, e)
        if attempt < max_retries:
            wait = retry_delay * attempt
            logger.info("等待 %.0f 秒后重试...", wait)
            time.sleep(wait)

    logger.error("拉取全 A 股行情最终失败（共重试 %d 次）: %s", max_retries, last_error)
    return None


def _fetch_history(code: str, days: int = 30) -> Optional["pd.DataFrame"]:
    """通过 efinance 拉取单只股票的日 K 线历史数据用于计算均线。"""
    import efinance as ef
    from datetime import datetime, timedelta

    beg = (datetime.now() - timedelta(days=days + 20)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    try:
        df = _ef_call_with_timeout(
            ef.stock.get_quote_history,
            stock_codes=code,
            beg=beg,
            end=end,
            klt=101,  # 日线
            fqt=1,    # 前复权
            timeout=60,
        )
        return df
    except Exception as e:
        logger.debug("拉取 %s 历史K线失败: %s", code, e)
        return None


def _normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    统一 efinance 返回的列名为内部统一名称。

    efinance 返回的列名：股票代码, 股票名称, 最新价, 涨跌幅, 涨跌额,
    成交量, 成交额, 振幅, 最高, 最低, 开盘, 换手率, 量比, 市盈率, 总市值, 流通市值
    """
    col_map = {
        "股票代码": "代码",
        "股票名称": "名称",
        "最新价": "最新价",
        "涨跌幅": "涨跌幅",
        "涨跌额": "涨跌额",
        "成交量": "成交量",
        "成交额": "成交额",
        "振幅": "振幅",
        "最高": "最高",
        "最低": "最低",
        "开盘": "开盘",
        "换手率": "换手率",
        "量比": "量比",
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def _screen_with_history(df_spot: "pd.DataFrame", max_count: int = 10) -> List[str]:
    """
    策略一（精确版）：逐只拉取历史 K 线计算均线。
    适用于候选股数量较少的场景。

    先用实时快照做粗筛，再对粗筛结果逐一拉历史 K 线计算均线确认。
    """
    import pandas as pd

    df = _normalize_columns(df_spot.copy())

    # 过滤 ST、退市
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", regex=True, na=False)]
    # 涨跌幅有效
    if "涨跌幅" in df.columns:
        df = df[df["涨跌幅"] != "-"]
        df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        df = df.dropna(subset=["涨跌幅"])
    else:
        logger.warning("缺少涨跌幅列，无法粗筛")
        return []

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

    利用"量比 > 1"作为资金关注度粗判，
    适合 GitHub Actions 时间受限的场景（默认使用此策略）。
    """
    import pandas as pd

    df = _normalize_columns(df_spot.copy())

    # 过滤 ST、退市
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", regex=True, na=False)]

    # 确保必要列存在
    if "涨跌幅" not in df.columns:
        logger.error("数据缺少涨跌幅列，无法筛选")
        return []

    df = df[df["涨跌幅"] != "-"]
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df.dropna(subset=["涨跌幅"])

    for col in ("成交量", "成交额", "换手率", "量比"):
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

    # 量比 > 0.8（有一定资金关注度），如果该列存在
    if "量比" in df.columns:
        cond = cond & (df["量比"] >= 0.8)

    # 换手率在合理区间（1%~15%），如果该列存在
    if "换手率" in df.columns:
        cond = cond & (df["换手率"] >= 1) & (df["换手率"] <= 15)

    screen_result = df[cond]

    # 按成交额降序
    if "成交额" in screen_result.columns:
        screen_result = screen_result.sort_values("成交额", ascending=False)

    stock_code_list = screen_result["代码"].head(max_count).tolist()
    return stock_code_list


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
    df = _fetch_realtime()

    if df is None:
        logger.error("无法获取行情数据，自动选股失败，将回退到默认 STOCK_LIST 配置")
        return ""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="A股自动选股")
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
