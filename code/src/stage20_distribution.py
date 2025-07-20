#%%
import pandas as pd
from auto_config import project_dir
df158 = pd.read_pickle(project_dir / project_dir / "temp/qlib_alpha158_ranked_with_stock_finance_info.pkl")
df158
#%%
# 先解决vwap不存在的问题。
# df158["VWAP0"] = (df158['Turnover'] / df158['volume'])/df158['close']


# %%
# 缺失值问题
# 将前60天数据删掉，估计就没有缺失值了
# df158['timestamp'].min()+60, df158['timestamp'].max()
# import missingno as msno
# msno.matrix(df158.sort_values('timestamp'))   # 按日期排序后看缺失分布
# TODO 详细分析能不能删除

# 这里直接粗暴删除
before = len(df158)
df158 = df158.dropna()
after = len(df158)
print(f"删除缺失值前后数据量变化: {before} -> {after}, delta: {before - after}")

# %%
# inf 问题
import numpy as np
# TODO
# 问题的根源是 前复权，股价变成负数，分红让涨跌幅突破20 10 限制。
# 处理 inf 和 -inf 值
numeric_cols = df158.select_dtypes(include=[float]).columns
for col in numeric_cols:
    if np.isinf(df158[col]).any():
        # 获取非inf值
        non_inf_values = df158[col][~np.isinf(df158[col])]
        if len(non_inf_values) > 0:
            col_max = non_inf_values.max()
            col_min = non_inf_values.min()
            # 将 inf 替换为最大值，-inf 替换为最小值
            df158.loc[:, col] = df158[col].replace([np.inf, -np.inf], [col_max, col_min])
np.isinf(df158.select_dtypes(include=[float])).any().any()  # 检查是否还有 inf 值

# %%
# 离群值问题
target_cols=['收盘', '涨跌幅', '涨跌幅排名', '涨跌正负', '涨跌', '龙虎']
variables = numeric_cols.difference(target_cols).difference(['timestamp', 'item_id']).tolist()
# '涨跌幅参与排名股票数量'
from feature_engine.outliers import Winsorizer
winsorizer = Winsorizer(
    capping_method='gaussian',
    tail='both',
    fold=3,
    variables=variables
)
df158_winsorized = winsorizer.fit_transform(df158)
# 检查是否还有离群值
winsorizer.right_tail_caps_
# %%
import pandas as pd
import numpy as np
from feature_engine.transformation import (
    YeoJohnsonTransformer,
    ArcsinTransformer,
    BoxCoxTransformer,
    ReciprocalTransformer,
    LogTransformer,
    LogCpTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. 构造映射表（key → 实例）
TRANSFORMERS = {
    'log':        LogTransformer(),
    'logcp':      LogCpTransformer(),
    'reciprocal': ReciprocalTransformer(),
    'boxcox':     BoxCoxTransformer(),
    'yeojohnson': YeoJohnsonTransformer(),
    'arcsin':     ArcsinTransformer(),
}

# 2. 自动选择变换器
def choose_transform(series):
    """
    根据分布特征返回最合适的变换器 key
    """
    s = series.dropna()
    if s.empty:
        return None
    skew = s.skew()
    all_pos = (s > 0).all()
    in_01   = (s >= 0).all() and (s <= 1).all()

    if in_01:
        return 'arcsin'
    if not all_pos:
        return 'yeojohnson'
    if abs(skew) < 0.5:
        return None
    if skew > 2:
        return 'reciprocal'
    if skew > 0.5:
        return 'log'
    return 'boxcox'

# 3. 主函数
def auto_transform(df, exclude_cols=None):
    """
    对 DataFrame 的数值列做自动变换
    返回：transformed_df, mapping_dict
    """
    exclude_cols = set(exclude_cols or [])
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)

    # 为每列挑选变换器，并构造 ColumnTransformer
    transformers = []
    mapping = {}

    for col in num_cols:
        key = choose_transform(df[col])
        if key is None:
            continue
        tf = TRANSFORMERS[key]
        transformers.append((f"{col}_{key}", tf, [col]))
        mapping[col] = key

    ct = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )

    transformed = ct.fit_transform(df)

    # 重新整理列名（保持原列顺序）
    new_cols = [t[2][0] for t in transformers]
    passthrough = [c for c in df.columns if c not in new_cols]
    new_cols.extend(passthrough)

    transformed_df = pd.DataFrame(transformed, columns=new_cols, index=df.index)
    return transformed_df, mapping




df158_winsorized[variables], used = auto_transform(df158_winsorized[variables])
print("变换器映射：", used)
df158_winsorized

#%%
import pandas as pd
import numpy as np
from scipy.stats import norm

# --- 核心功能函式 (可被其他檔案匯入) ---

def quantile_transform_column(column: pd.Series) -> pd.Series:
    """
    對 Pandas Series (DataFrame 的一個欄位) 進行分位數變換，
    使其服從標準高斯分佈 (均值為0，方差為1)。

    這個變換會保持原始數據的相對大小關係 (rank-preserving)。

    Args:
        column: 一個 Pandas Series 物件，包含數值型數據。

    Returns:
        一個新的 Pandas Series，其值服從標準高斯分佈。
        原始數據中的 NaN 值會被保留。
    """
    # 1. 計算每個值在欄位中的排名。'average' 方法可以妥善處理相同值。
    ranks = column.rank(method='average')
    
    # 2. 計算非缺失值的總數 N
    n_total = column.count()
    
    # 3. 將排名轉換為 (0, 1) 區間內的分位數。
    #    分母使用 n_total + 1 是為了避免產生 1.0 的分位數，
    #    因為 norm.ppf(1.0) 會得到無窮大。
    quantiles = ranks / (n_total + 1)
    
    # 4. 使用 SciPy 的 norm.ppf (標準高斯分佈的百分點函數)
    #    將分位數映射到高斯分佈上。
    transformed_series = norm.ppf(quantiles)
    
    return transformed_series
# df158['相对涨跌幅'] = quantile_transform_column(df158['涨跌幅'])
df158_winsorized['涨跌幅排名'] = df158_winsorized['涨跌幅排名'].astype(float)  # 确保是浮点数类型
df158_winsorized.loc[:, '涨跌幅排名'] = quantile_transform_column(df158_winsorized['涨跌幅排名'])
# TODO
# 收盘价可以做可逆变换。

# %%
df158_winsorized.to_pickle(project_dir / "temp/qlib_alpha158_finance_winsorized.pkl")
# %%
df158_winsorized
# %%
