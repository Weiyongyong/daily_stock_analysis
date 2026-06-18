# -*- coding: utf-8 -*-
"""
自动选股模块

通过 AkShare 拉取全 A 股实时行情，并结合历史 K 线计算均线，
按"均线多头排列 + 放量上涨 + 涨跌幅温和"策略筛选候选股。

用法:
    python auto_screen.py            # 筛选并打印结果，同时写入 .auto_stock_env
    python auto_screen.py --max 5     # 最多筛选 5 只

在 GitHub Actions 中，本脚本在 main.py 之前运行，筛选结果通过
环境变量 STOCK_LIST 直接传递给 main.py（同时写入 .auto_stock_env 供本地使用）。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger("auto_screen")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _compute_ma(df, col: str, window: int) -> "pd.Series":
    """在 DataFrame 上计算指定列的简单移动平均。"""
    return df[col].rolling(window=window, min_periods=1).mean()


def _fetch_spot() -> "pd.DataFrame":
    """拉取全 A 股实时快照行情。"""
    import akshare as ak

    logger.info("正在拉取全 A 股实时行情 (ak.stock_zh_a_spot_em) ...")
    df = ak.stock_zh_a_spot_em()
    logger.info("拉取完成，共 %d 条记录", len(df))
    return df


def _fetch_history(code: str, days: int = 30) -> "pd.DataFrame":
    """拉取单只股票的日 K 线历史数据用于计算均线。"""
    import akshare as ak

    from datetime import datetime, timedelta

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 20)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    return df


def _screen_with_history(df_spot: "pd.DataFrame", max_count: int = 10) -> List[str]:
    """
    策略一（精确版）：逐只拉取历史 K 线计算均线。
    适用于候选股数量较少的场景。

    先用实时快照做粗筛，再对粗筛结果逐一拉历史 K 线计算均线确认。
    """
    import pandas as pd

    # --- 粗筛：用实时快照字段 ---
    df = df_spot.copy()

    # 过滤 ST、退市
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", regex=True, na=False)]
    # 涨跌幅有效
    df = df[df["涨跌幅"] != "-"]
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df.dropna(subset=["涨跌幅"])

    # 粗筛条件：涨跌幅温和（不追高）
    rough = df[(df["涨跌幅"] >= -2) & (df["涨跌幅"] <= 5)]
    # 按成交额降序，优先关注活跃股
    if "成交额" in rough.columns:
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

    利用"60日涨跌幅 > 0"作为趋势向上一级粗判，
    适合 GitHub Actions 时间受限的场景（默认使用此策略）。
    """
    import pandas as pd

    df = df_spot.copy()

    # 过滤 ST、退市
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", regex=True, na=False)]
    df = df[df["涨跌幅"] != "-"]
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df.dropna(subset=["涨跌幅"])

    if "成交量" in df.columns:
        df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce")
    if "成交额" in df.columns:
        df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
    if "换手率" in df.columns:
        df["换手率"] = pd.to_numeric(df["换手率"], errors="coerce")
    if "60日涨跌幅" in df.columns:
        df["60日涨跌幅"] = pd.to_numeric(df["60日涨跌幅"], errors="coerce")

    # 构建筛选条件
    cond = (
        # 涨跌幅温和，不追高
        (df["涨跌幅"] >= -2)
        & (df["涨跌幅"] <= 5)
        # 成交量 > 0（排除停牌）
        & (df["成交量"] > 0)
    )

    # 60日涨跌幅为正（中期趋势向上），如果该列存在
    if "60日涨跌幅" in df.columns:
        cond = cond & (df["60日涨跌幅"] > 0)

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
        逗号分隔的股票代码字符串，如 "600519,000001"
    """
    df = _fetch_spot()

    if use_history:
        stock_code_list = _screen_with_history(df, max_count=max_count)
    else:
        stock_code_list = _screen_quick(df, max_count=max_count)

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
        print("未筛选到符合条件的股票")
        return 1


if __name__ == "__main__":
    sys.exit(main())
