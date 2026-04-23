import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def get_fund_data(fund_code):
    try:
        # 获取对应基金数据
        df = ak.fund_money_fund_info_em(fund_code)
        if df is None or df.empty:
            return None
        
        # 数据预处理
        df.columns = ['净值日期', '每万份收益', '7日年化收益率', '申购状态', '赎回状态']    # 重命名DF列名        
        df['净值日期'] = pd.to_datetime(df['净值日期'])  # 转换为PD日期时间类型
        df['每万份收益'] = pd.to_numeric(df['每万份收益'], errors='coerce')     # 转换为PD数值类型
        df = df.dropna(subset=['每万份收益'])       # 删除在 每万份收益 这一列上为缺失值（NaN）的行
        df = df.sort_values('净值日期').reset_index(drop=True)  # 按净值日期排序并重置索引

        # 选择最近一年数据
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        df = df[(df['净值日期'] >= one_year_ago) & (df['净值日期'] <= today)].copy()
        if df.empty:
            return None

        # 计算年化收益率
        df['累计每万份收益'] = df['每万份收益'].cumsum()
        df['年化收益率'] = df['累计每万份收益'] / 100
        return df
    except Exception as e:
        print(f"异常: {type(e).__name__}: {e}")
        return None

def main():
    codes_input = input("请输入基金代码（逗号分隔）：").strip()
    if not codes_input:
        print("错误：未输入基金代码")
        return

    fund_codes = [c.strip() for c in codes_input.split(",") if c.strip()]
    if not fund_codes:
        print("错误：未输入有效的基金代码")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for code in fund_codes:
        print(f"正在获取 {code} ...")
        df = get_fund_data(code)
        if df is None or df.empty:
            print(f"{code} 数据获取失败")
            continue

        ax.plot(df['净值日期'], df['年化收益率'], label=code, linewidth=1.5)
        latest = df.iloc[-1]
        print(f"{code}: 最新日期 {latest['净值日期'].strftime('%Y-%m-%d')}, "
              f"每万份收益 {latest['每万份收益']:.4f}, "
              f"年化收益率 {latest['年化收益率']:.4f}%")

    ax.set_title("货币基金年化收益率走势", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("年化收益率 (%)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # print("AKshare安装成功, 版本号:", ak.__version__)
    main()