from datetime import datetime, timedelta

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd

import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Database')  # 缓存文件夹路径
CACHE_EXPIRE_DAYS = 1  # 缓存有效期（天）

def save_fund_cache(fund_code, df):
    """保存基金数据到缓存文件"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)  # 创建缓存目录（如果不存在）
    cache_file = os.path.join(CACHE_DIR, f'{fund_code}.pkl')
    cache_data = {
        'df': df,  # 基金数据
        'cache_time': datetime.now()  # 缓存时间
    }
    pd.to_pickle(cache_data, cache_file)  # 使用pandas序列化保存

def load_fund_cache(fund_code):
    """从缓存文件加载基金数据，若缓存不存在或已过期则返回None"""
    cache_file = os.path.join(CACHE_DIR, f'{fund_code}.pkl')
    if not os.path.exists(cache_file):
        return None  # 缓存文件不存在
    
    cache_data = pd.read_pickle(cache_file)  # 反序列化读取缓存
    cache_time = cache_data['cache_time']  # 获取缓存保存时间
    # 检查缓存是否过期：当前时间 - 缓存时间 > 缓存有效期
    if (datetime.now() - cache_time).days >= CACHE_EXPIRE_DAYS:
        return None  # 缓存已过期
    
    return cache_data['df']  # 返回缓存的基金数据

def get_fund_data(fund_code):
    # 尝试从缓存加载数据
    df = load_fund_cache(fund_code)
    if df is not None:
        print(f"{fund_code}: 从缓存加载数据")
        return df
    
    try:
        # 获取对应基金数据
        df = ak.fund_money_fund_info_em(fund_code)
        if df is None or df.empty:
            return None
        
        # 数据预处理
        df.columns = ['净值日期', '每万份收益', '7日年化收益率', '申购状态', '赎回状态']    # 重命名DF列名        
        df['净值日期'] = pd.to_datetime(df['净值日期'], errors='coerce')  # 转换为PD日期时间类型，解析失败则为NaT
        df['每万份收益'] = pd.to_numeric(df['每万份收益'], errors='coerce')     # 转换为PD数值类型
        df = df.dropna(subset=['净值日期', '每万份收益'])       # 删除在 净值日期 或 每万份收益 上为缺失值的行
        df = df.sort_values('净值日期').reset_index(drop=True)  # 按净值日期排序并重置索引

        # 选择最近一年数据
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        df = df[(df['净值日期'] >= one_year_ago) & (df['净值日期'] <= today)].copy()
        if df.empty:
            return None

        # 复利方式计算年化收益率
        df['年化收益率'] = ((1 + df['每万份收益'] / 10000).cumprod() - 1) * 100

        # 保存数据到缓存
        save_fund_cache(fund_code, df)
        print(f"{fund_code}: 从API获取数据并更新缓存")
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

    fig, ax = plt.subplots(figsize=(12, 8))

    for code in fund_codes:
        print(f"正在获取 {code} ...")
        df = get_fund_data(code)
        if df is None or df.empty:
            print(f"{code} 数据获取失败")
            continue

        ax.plot(df['净值日期'], df['年化收益率'], label=code, linewidth=1.5)
        latest = df.iloc[-1]
        # 在最新数据点标注年化收益率数值
        ax.annotate(f'{latest["年化收益率"]:.3f}%', 
                    xy=(latest['净值日期'], latest['年化收益率']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))
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