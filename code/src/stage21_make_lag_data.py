# %%
import pandas as pd
from typing import List, Optional

class TimeSeriesLagPreprocessor:
    def __init__(self, lag_days: Optional[List[int]] = None):
        if lag_days is None:
            self.lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 56, 84]
            self.lag_days = [x - 1 for x in self.lag_days[1:]]  # 去掉1天，减1避免周期偏移
        else:
            self.lag_days = lag_days

    def preprocess(self, df: pd.DataFrame, 
                   target_cols: Optional[List[str]] = None, 
                   lag_target_only: bool = False) -> pd.DataFrame:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

        # 1️⃣ 填充完整日期
        filled_list = []
        for item_id, group in df_sorted.groupby('item_id'):
            full_range = pd.date_range(group['timestamp'].min(), group['timestamp'].max(), freq='D')
            full_df = pd.DataFrame({'timestamp': full_range})
            full_df['item_id'] = item_id
            merged = pd.merge(full_df, group, on=['item_id', 'timestamp'], how='left', sort=True)
            filled_list.append(merged)
        df_filled = pd.concat(filled_list, ignore_index=True)

        # 2️⃣ 做 lag 特征
        if lag_target_only and target_cols is not None:
            cols_to_lag = target_cols
        else:
            cols_to_lag = [col for col in df_filled.columns if col not in ['item_id', 'timestamp']]

        lag_features = []
        for col in cols_to_lag:
            for lag in self.lag_days:
                lag_col = df_filled.groupby('item_id')[col].shift(lag)
                lag_col.name = f"{col}_lag_{lag}"
                lag_features.append(lag_col)
        if lag_features:
            lag_df = pd.concat(lag_features, axis=1)
            df_filled = pd.concat([df_filled, lag_df], axis=1)

        # 3️⃣ 删除当第5-10列（索引4-9）全是NaN的行
        cols_to_check = df_filled.columns[4:10]
        mask_all_nan = df_filled[cols_to_check].isna().all(axis=1)
        df_filled = df_filled[~mask_all_nan].reset_index(drop=True)


        return df_filled

def add_shift_features(df: pd.DataFrame,
                       target_cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    # 确保 target 列在末尾
    for col in target_cols:
        col_data = df.pop(col)
        df[col] = col_data

    shift_features = []
    for col in target_cols:
        shift_col = df.groupby('item_id')[col].shift(-1)
        shift_col.name = f"{col}_shift"
        shift_features.append(shift_col)
    if shift_features:
        shift_df = pd.concat(shift_features, axis=1)
        df = pd.concat([df, shift_df], axis=1)
    return df

# %%
from auto_config import project_dir

input_file = project_dir / "temp/qlib_alpha158_finance_winsorized.pkl"
output_file = project_dir / "temp/lag158_finance.pkl"

# input_file = project_dir / "temp/qlib_alpha158_winsorized.pkl"
# output_file = project_dir / "temp/lag158.pkl"

df_raw = pd.read_pickle(input_file)
unuseful_cols = ['涨跌幅参与排名股票数量']
df_raw = df_raw.drop(columns=unuseful_cols)

# 1️⃣ lag 特征 + 删除全空行
preprocessor = TimeSeriesLagPreprocessor()
df_lagged = preprocessor.preprocess(
    df_raw,
    target_cols=['收盘', '涨跌幅', '涨跌幅排名', '涨跌正负', '涨跌', '龙虎'],
    lag_target_only=True
)

# 2️⃣ 添加 shift 特征
df_shifted = add_shift_features(
    df_lagged,
    target_cols=['收盘', '涨跌幅', '涨跌幅排名', '涨跌正负', '涨跌', '龙虎']
)


df_shifted.to_pickle(output_file)
print(f"✅ Saved cleaned file to {output_file}")

# %%
df_shifted
# %%
