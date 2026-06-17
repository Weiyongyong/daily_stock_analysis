import akshare as ak
import os

def auto_screen_stocks():
    # 1. 拉取全A股实时行情
    df = ak.stock_zh_a_spot_em()
    # 过滤ST、退市、停牌、异常股
    df = df[~df["名称"].str.contains("ST|*ST|退")]
    df = df[df["涨跌幅"] != "-"]
    df["涨跌幅"] = df["涨跌幅"].astype(float)
    df["成交量"] = df["成交量"].astype(float)
    df["20日均量"] = df["20日均量"].astype(float)
    df["MA5"] = df["MA5"].astype(float)
    df["MA10"] = df["MA10"].astype(float)
    df["MA20"] = df["MA20"].astype(float)

    # ========== 短线多头选股条件，可自行修改 ==========
    filter_condition = (
        # 均线多头排列
        (df["MA5"] > df["MA10"]) &
        (df["MA10"] > df["MA20"]) &
        # 成交量大于20日均量，放量上涨
        (df["成交量"] > df["20日均量"]) &
        # 当日涨跌幅温和，不追高
        (df["涨跌幅"] >= -2) &
        (df["涨跌幅"] <= 5)
    )
    # ==============================================

    screen_result = df[filter_condition]
    # 最多筛选10只，防止大模型调用超限
    stock_code_list = screen_result["代码"].head(10).tolist()
    stock_str = ",".join(stock_code_list)
    print(f"今日自动筛选股票池：{stock_str}")

    # 将选股结果写入临时环境文件，供main.py读取
    with open(".auto_stock_env", "w", encoding="utf-8") as f:
        f.write(f"STOCK_LIST={stock_str}")
    return stock_str

if __name__ == "__main__":
    auto_screen_stocks()
