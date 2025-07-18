#%%
import pandas as pd
from typing import List, Optional
import sys
import os

class TimeSeriesLagPreprocessor:
    def __init__(self, lag_days: Optional[List[int]] = None):
        if lag_days is None:
            self.lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 56, 84]
        else:
            self.lag_days = lag_days

    def preprocess(self, df: pd.DataFrame, drop_last_n: int = 2, 
                   target_cols: Optional[List[str]] = None, 
                   dropna_shift_targets: bool = True) -> pd.DataFrame:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
        filled_list = []
        for item_id, group in df_sorted.groupby('item_id'):
            full_range = pd.date_range(group['timestamp'].min(), group['timestamp'].max(), freq='D')
            full_df = pd.DataFrame({'timestamp': full_range})
            full_df['item_id'] = item_id
            merged = pd.merge(full_df, group, on=['item_id', 'timestamp'], how='left', sort=True)
            filled_list.append(merged)
        df_filled = pd.concat(filled_list, ignore_index=True)
        # cols_to_lag = df_filled.columns[2:-drop_last_n] if drop_last_n > 0 else df_filled.columns[2:]
        cols_to_lag = df_filled.columns[2:]  # 去除前两列，但包含剩下所有列

        # 使用批量拼接方式避免碎片化
        lag_features = []
        for col in cols_to_lag:
            for lag in self.lag_days:
                lag_col = df_filled.groupby('item_id')[col].shift(lag)
                lag_col.name = f'{col}_lag_{lag}'
                lag_features.append(lag_col)
        if lag_features:
            lag_df = pd.concat(lag_features, axis=1)
            df_filled = pd.concat([df_filled, lag_df], axis=1)

        if target_cols is not None:
            for col in target_cols:
                col_data = df_filled.pop(col)
                df_filled[col] = col_data

            # 批量拼接 shift 列避免碎片化
            shift_features = []
            for col in target_cols:
                shift_col = df_filled.groupby('item_id')[col].shift(-1)
                shift_col.name = f'{col}_shift'
                shift_features.append(shift_col)
            if shift_features:
                shift_df = pd.concat(shift_features, axis=1)
                df_filled = pd.concat([df_filled, shift_df], axis=1)

        if df_filled.shape[1] >= 7:
            cols_check = df_filled.columns[3:7]
            mask_all_nan = df_filled[cols_check].isna().all(axis=1)
            df_filled = df_filled[~mask_all_nan].reset_index(drop=True)
        if dropna_shift_targets and target_cols is not None:
            shift_cols = [f'{col}_shift' for col in target_cols]
            df_filled = df_filled.dropna(subset=shift_cols).reset_index(drop=True)
        return df_filled

# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print(f"Usage: python {os.path.basename(__file__)} input_file.csv output_file.csv")
#         sys.exit(1)

    # input_file = sys.argv[1]
    # output_file = sys.argv[2]
#%%
from auto_config import project_dir
input_file = project_dir / "temp/qlib_alpha158_ranked_with_stock_info.pkl"
output_file = project_dir / "temp/lag158.pkl"


df_raw = pd.read_pickle(input_file)

unuseful_cols = [
    '涨跌幅参与排名股票数量',
]
df_raw = df_raw.drop(columns=unuseful_cols)

preprocessor = TimeSeriesLagPreprocessor()
df_cleaned = preprocessor.preprocess(
    df_raw,
    target_cols=['收盘', '涨跌幅', '涨跌幅排名', '涨跌正负', '涨跌', '龙虎']
)
df_cleaned.to_pickle(output_file)
print(f"Saved cleaned file to {output_file}")
#%%
