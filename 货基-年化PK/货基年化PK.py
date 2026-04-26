import os
import warnings
os.environ['TQDM_DISABLE'] = '1'  # 全局禁用tqdm进度条
os.environ['PYCHARM_HOSTED'] = '1'  # 避免 tqdm 在某些环境下的特殊处理

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd

try:
    import tqdm
    original_init = tqdm.tqdm.__init__
    def disabled_init(self, *args, **kwargs):
        kwargs['disable'] = kwargs.get('disable', True)
        return original_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = disabled_init
except (ImportError, AttributeError):
    pass

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 屏蔽akshare日期解析警告
warnings.filterwarnings('ignore', category=UserWarning, module='akshare')  

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Database')  # 缓存文件夹路径
CACHE_EXPIRE_DAYS = 1  # 缓存有效期（天）
MAX_THREADS = 10  # 最大线程数
BLACKLIST_FILE = os.path.join(CACHE_DIR, 'blacklist.txt')  # 黑名单文件路径

def load_fund_blacklist():
    """从黑名单文件加载基金代码列表，文件不存在或读取失败时返回空集合"""
    if not os.path.exists(BLACKLIST_FILE):
        return set()  # 黑名单文件不存在，返回空集合
    try:
        with open(BLACKLIST_FILE, 'r', encoding='utf-8') as f:
            # 读取所有行，去除空白字符，跳过空行
            blacklist = {line.strip() for line in f if line.strip()}
        return blacklist
    except Exception as e:
        print(f"警告: 读取黑名单文件失败: {type(e).__name__}: {e}")
        return set()  # 读取失败，返回空集合

def load_fund_cache(fund_code):
    """从缓存文件加载基金数据，若缓存不存在或已过期则返回None"""
    cache_file = os.path.join(CACHE_DIR, f'{fund_code}.pkl')
    if not os.path.exists(cache_file):
        return None  # 缓存文件不存在
    
    cache_data = pd.read_pickle(cache_file)  # 反序列化读取缓存
    cache_time = cache_data['cache_time']  # 获取缓存保存时间
    # 如果缓存保存日期 < 当前日期，则认为已过期
    if cache_time.date() < datetime.now().date():
        return None  # 缓存已过期
    
    return cache_data['df']  # 返回缓存的基金数据

def save_fund_cache(fund_code, df):
    """保存基金数据到缓存文件"""
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)  # 创建缓存目录（如果不存在）
    cache_file = os.path.join(CACHE_DIR, f'{fund_code}.pkl')
    cache_data = {
        'df': df,  # 基金数据
        'cache_time': datetime.now()  # 缓存时间
    }
    pd.to_pickle(cache_data, cache_file)  # 使用pandas序列化保存

def get_fund_data(fund_code):
    # 检查是否在黑名单中
    blacklist = load_fund_blacklist()
    if fund_code in blacklist:
        print(f"{fund_code}: 该基金在黑名单中，已跳过")
        return None
    
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
        
        # 过滤异常数据：年化收益率超过100%的基金可能是数据异常（如万元收益实际是净值）
        max_reasonable_rate = 100.0  # 货币基金年化收益率正常范围约1%-5%，上限设为100%
        if df['年化收益率'].iloc[-1] > max_reasonable_rate:
            print(f"{fund_code}: 数据异常（年化收益率 {df['年化收益率'].iloc[-1]:.2f}%）")
            return None

        # 保存数据到缓存
        save_fund_cache(fund_code, df)
        print(f"{fund_code}: 从API获取数据并更新缓存")
        return df
    except Exception as e:
        print(f"异常: {type(e).__name__}: {e}")
        return None

def fetch_fund_rate(code):
    """多线程获取单只基金数据，用于update_mode中的并行获取"""
    df = get_fund_data(code)
    if df is None or df.empty:
        return None
    latest = df.iloc[-1]
    return {
        'code': code,
        'date': latest['净值日期'],
        'rate': latest['年化收益率'],
        'df': df
    }

def update_mode():
    """更新模式：获取所有货币基金列表，选取年化收益率最高的前5位对比显示"""
    print("正在获取所有货币基金列表...")
    try:
        all_funds = ak.fund_money_fund_daily_em()  # 获取所有货币基金列表
        if all_funds is None or all_funds.empty:
            print("错误：无法获取货币基金列表")
            return
        fund_codes = all_funds['基金代码'].tolist()  # 提取基金代码列表
        print(f"发现 {len(fund_codes)} 只货币基金，开始获取年化收益率...")
    except Exception as e:
        print(f"异常: {type(e).__name__}: {e}")
        return
    
    fund_rates = []
    count = 0
    
    # 使用多线程并行获取基金数据
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_code = {executor.submit(fetch_fund_rate, code): code for code in fund_codes}
        for future in as_completed(future_to_code):
            result = future.result()
            if result is not None:
                fund_rates.append(result)
            count += 1
            if count % 50 == 0:
                print(f"已处理 {count} 只基金...")
    
    if not fund_rates:
        print("错误：无可用基金数据")
        return
    
    fund_rates.sort(key=lambda x: x['rate'], reverse=True)
    top5 = fund_rates[:5]
    
    print("\n=== 货币基金年化收益率排名前5 ===")
    for i, fund in enumerate(top5, 1):
        print(f"第{i}名: {fund['code']} | 年化收益率: {fund['rate']:.4f}% | 最新日期: {fund['date'].strftime('%Y-%m-%d')}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, fund in enumerate(top5):
        ax.plot(fund['df']['净值日期'], fund['df']['年化收益率'], 
                label=fund['code'], linewidth=1.5, color=colors[i])
        ax.annotate(f"{fund['rate']:.3f}%", 
                    xy=(fund['date'], fund['rate']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color=colors[i],
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=0.5))
    
    ax.set_title("货币基金年化收益率 TOP5 走势", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("年化收益率 (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 40)
    print("货币基金年化收益率对比工具")
    print("=" * 40)
    print("1. 自填模式：手动输入基金代码进行对比")
    print("2. 更新模式：自动获取全市场货币基金，取年化收益率前5名")
    print("=" * 40)
    
    mode = input("请选择模式（1/2）：").strip()
    
    if mode == '1':
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
    elif mode == '2':
        update_mode()
    else:
        print("无效的选择，请输入 1 或 2")

if __name__ == "__main__":
    # print("AKshare安装成功, 版本号:", ak.__version__)
    main()