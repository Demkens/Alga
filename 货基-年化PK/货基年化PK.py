import os
import warnings

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TQDM_DISABLE'] = '1'
os.environ['PYCHARM_HOSTED'] = '1'

# 屏蔽tqdm进度条的disable参数
try:
    import tqdm
    original_init = tqdm.tqdm.__init__
    def disabled_init(self, *args, **kwargs):
        kwargs['disable'] = kwargs.get('disable', True)
        return original_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = disabled_init
except (ImportError, AttributeError):
    pass

warnings.filterwarnings('ignore', category=UserWarning, module='akshare')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class CacheData:
    """缓存数据结构，包含基金数据和时间戳"""
    df: pd.DataFrame
    cache_time: datetime

@dataclass
class FundRecord:
    """单只基金记录数据结构"""
    code: str
    date: datetime
    rate: float
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

class FundDataFetcher:
    """货币基金数据获取器，封装数据获取、缓存和黑名单管理"""
    def __init__(self,
        cache_dir: str,
        cache_expire_days: int = 1,
        max_threads: int = 10,
        blacklist_file: Optional[str] = None
    ):
        self.cache_dir = cache_dir
        self.cache_expire_days = cache_expire_days
        self.max_threads = max_threads
        self.blacklist_file = blacklist_file or os.path.join(cache_dir, 'blacklist.txt')
        self._blacklist: Optional[set] = None

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _load_blacklist(self) -> set:
        """加载黑名单列表"""
        if self._blacklist is not None:
            return self._blacklist

        if not os.path.exists(self.blacklist_file):
            self._blacklist = set()
            return self._blacklist

        try:
            with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                self._blacklist = {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"警告: 读取黑名单文件失败: {type(e).__name__}: {e}")
            self._blacklist = set()
        return self._blacklist

    def is_blacklisted(self, fund_code: str) -> bool:
        """检查基金是否在黑名单中"""
        return fund_code in self._load_blacklist()

    def _load_cache(self, fund_code: str) -> Optional[pd.DataFrame]:
        """从缓存加载基金数据"""
        cache_file = os.path.join(self.cache_dir, f'{fund_code}.pkl')
        if not os.path.exists(cache_file):
            return None

        try:
            cache_data: CacheData = pd.read_pickle(cache_file)
            if cache_data.cache_time.date() < datetime.now().date():
                return None
            return cache_data.df
        except Exception:
            return None

    def _save_cache(self, fund_code: str, df: pd.DataFrame) -> None:
        """保存基金数据到缓存"""
        self._ensure_cache_dir()
        cache_file = os.path.join(self.cache_dir, f'{fund_code}.pkl')
        cache_data = CacheData(df=df, cache_time=datetime.now())
        pd.to_pickle(cache_data, cache_file)

    def _process_fund_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """预处理基金数据：列重命名、类型转换、筛选最近一年、计算年化收益率"""
        if df is None or df.empty:
            return None

        df.columns = ['净值日期', '每万份收益', '7日年化收益率', '申购状态', '赎回状态']
        df['净值日期'] = pd.to_datetime(df['净值日期'], errors='coerce')
        df['每万份收益'] = pd.to_numeric(df['每万份收益'], errors='coerce')
        df = df.dropna(subset=['净值日期', '每万份收益'])
        df = df.sort_values('净值日期').reset_index(drop=True)

        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        df = df[(df['净值日期'] >= one_year_ago) & (df['净值日期'] <= today)].copy()

        if df.empty:
            return None

        df['年化收益率'] = ((1 + df['每万份收益'] / 10000).cumprod() - 1) * 100
        return df

    def fetch_single(self, fund_code: str, use_cache: bool = True) -> Optional[FundRecord]:
        """获取单只基金数据"""
        if self.is_blacklisted(fund_code):
            print(f"{fund_code}: 该基金在黑名单中，已跳过")
            return None

        if use_cache:
            df = self._load_cache(fund_code)
            if df is not None:
                print(f"{fund_code}: 从缓存加载数据")
                latest = df.iloc[-1]
                return FundRecord(
                    code=fund_code,
                    date=latest['净值日期'],
                    rate=latest['年化收益率'],
                    df=df
                )

        try:
            raw_df = ak.fund_money_fund_info_em(fund_code)
            df = self._process_fund_data(raw_df)
            if df is None or df.empty:
                print(f"{fund_code}: 数据为空或处理后无有效数据")
                return None

            self._save_cache(fund_code, df)
            print(f"{fund_code}: 从API获取数据并更新缓存")
            latest = df.iloc[-1]
            return FundRecord(
                code=fund_code,
                date=latest['净值日期'],
                rate=latest['年化收益率'],
                df=df
            )
        except Exception as e:
            print(f"异常: {type(e).__name__}: {e}")
            return None

    def fetch_batch(self, fund_codes: list) -> list:
        """多线程批量获取基金数据"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self.fetch_single, code): code for code in fund_codes}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        return results

def plot_funds(
    funds: list,
    title: str,
    figsize: tuple = (12, 8),
    colors: Optional[list] = None
) -> None:
    """绘制基金年化收益率走势图的公共函数

    Args:
        funds: 基金记录列表
        title: 图表标题
        figsize: 图形尺寸
        colors: 颜色列表，默认使用内置颜色
    """
    if not funds:
        print("错误：无数据可绘制")
        return

    fig, ax = plt.subplots(figsize=figsize)
    default_colors = ['red', 'blue', 'green', 'orange', 'purple']
    plot_colors = colors or default_colors[:len(funds)]

    for i, fund in enumerate(funds):
        color = plot_colors[i % len(plot_colors)]
        ax.plot(fund.df['净值日期'], fund.df['年化收益率'],
                label=fund.code, linewidth=1.5, color=color)
        ax.annotate(
            f"{fund.rate:.3f}%",
            xy=(fund.date, fund.rate),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=0.5)
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("年化收益率 (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def update_mode(fetcher: FundDataFetcher) -> None:
    """更新模式：获取所有货币基金列表，选取年化收益率最高的前5位对比显示"""
    print("正在获取所有货币基金列表...")
    try:
        all_funds = ak.fund_money_fund_daily_em()
        if all_funds is None or all_funds.empty:
            print("错误：无法获取货币基金列表")
            return
        fund_codes = all_funds['基金代码'].tolist()
        print(f"发现 {len(fund_codes)} 只货币基金，开始获取年化收益率...")
    except Exception as e:
        print(f"异常: {type(e).__name__}: {e}")
        return

    fund_rates = fetcher.fetch_batch(fund_codes)
    if not fund_rates:
        print("错误：无可用基金数据")
        return

    fund_rates.sort(key=lambda x: x.rate, reverse=True)
    top5 = fund_rates[:5]

    print("\n=== 货币基金年化收益率排名前5 ===")
    for i, fund in enumerate(top5, 1):
        print(f"第{i}名: {fund.code} | 年化收益率: {fund.rate:.4f}% | 最新日期: {fund.date.strftime('%Y-%m-%d')}")

    plot_funds(top5, "货币基金年化收益率 TOP5 走势")

def main():
    """主函数：货币基金年化收益率对比工具入口"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fetcher = FundDataFetcher(
        cache_dir=os.path.join(script_dir, 'Database'),
        cache_expire_days=1,
        max_threads=10
    )

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

        funds = []
        for code in fund_codes:
            print(f"正在获取 {code} ...")
            fund = fetcher.fetch_single(code)
            if fund is not None:
                funds.append(fund)
                print(f"{code}: 最新日期 {fund.date.strftime('%Y-%m-%d')}, "
                      f"每万份收益 {fund.df.iloc[-1]['每万份收益']:.4f}, "
                      f"年化收益率 {fund.rate:.4f}%")
            else:
                print(f"{code}: 数据获取失败")

        if funds:
            plot_funds(funds, "货币基金年化收益率走势")
        else:
            print("错误：没有任何基金数据获取成功")
    elif mode == '2':
        update_mode(fetcher)
    else:
        print("无效的选择，请输入 1 或 2")

if __name__ == "__main__":
    main()