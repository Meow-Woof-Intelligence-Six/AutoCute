#%%
import pandas as pd
from pathlib import Path
from auto_config import project_dir, qlib_dir, custom_data_dir, valid_dates, train_dates
train_dates
#%%


# =============== 1️⃣ 读取沪深300成分股（去前导0） ===============
hs300_df = pd.read_csv(
    qlib_dir / "instruments/all.txt",
    sep='\t',
    header=None,
    names=['code', 'start_date', 'end_date']
)
hs300 = hs300_df['code'].astype(str).str.lstrip('0').tolist()

# =============== 2️⃣ 读取财务三表 ===============
Assets_Liabilities_df = pd.read_csv(custom_data_dir / "finance/Assets_Liabilities.csv")
Financial_df          = pd.read_csv(custom_data_dir / "finance/Financial.csv")
profit_df             = pd.read_csv(custom_data_dir / "finance/profit.csv")

# =============== 3️⃣ 清理列、去前导0、过滤沪深300 ===============
for df in (Assets_Liabilities_df, Financial_df, profit_df):
    df['股票代码'] = df['股票代码'].astype(str).str.lstrip('0')

Assets_Liabilities_df = Assets_Liabilities_df[Assets_Liabilities_df['股票代码'].isin(hs300)].reset_index(drop=True)
Financial_df          = Financial_df[Financial_df['股票代码'].isin(hs300)].reset_index(drop=True)
profit_df             = profit_df[profit_df['股票代码'].isin(hs300)].reset_index(drop=True)

# =============== 4️⃣ 重命名列统一格式 ===============
for df in (Assets_Liabilities_df, Financial_df, profit_df):
    if '序号' in df.columns:
        df.drop(columns=['序号'], inplace=True)
    df.rename(columns={'股票代码': 'item_id', 'date': 'timestamp'}, inplace=True)

# =============== 5️⃣ 合并三张财报表 ===============
df_financial = (
    Assets_Liabilities_df
    .merge(Financial_df, on=['item_id', 'timestamp'], how='outer', suffixes=('_al', '_fin'))
    .merge(profit_df, on=['item_id', 'timestamp'], how='outer')
)

# =============== 6️⃣ 财报表补齐完整日期 & 次日生效 ===============
def fill_financial_complete_date(df, start_date=train_dates[0], end_date=valid_dates[1]):
    # 转换日期格式
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d', errors='coerce')
    # 🚩 财报发布日次日才生效
    df['timestamp'] += pd.Timedelta(days=1)

    def fill_group(g):
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        g = g.set_index('timestamp').sort_index()
        g = g.reindex(full_dates)

        # 填充 item_id
        g['item_id'] = g['item_id'].ffill().bfill()

        # 填充股票简称、所处行业（若存在）
        for col in ['股票简称', '所处行业']:
            if col in g.columns:
                g[col] = g[col].ffill().bfill()

        # 填充其他财务列（forward fill）
        fill_cols = [c for c in g.columns if c not in ['item_id', '股票简称', '所处行业']]
        g[fill_cols] = g[fill_cols].ffill()

        g = g.reset_index().rename(columns={'index': 'timestamp'})
        g['timestamp'] = g['timestamp'].dt.strftime('%Y%m%d')
        return g

    df_filled = df.groupby('item_id', group_keys=False).apply(fill_group)
    return df_filled

df_financial_filled = fill_financial_complete_date(df_financial)

# =============== 7️⃣ 保存财报表填充后中间结果（可选） ===============
df_financial_filled.to_csv(project_dir / "temp/stage1/df_financial_filled.csv", index=False, encoding='utf-8-sig')

# =============== 8️⃣ 读取并填充 df158 完整日期 ===============
df158 = pd.read_pickle(project_dir / "temp/qlib_alpha158_ranked_with_stock_info.pkl")

def fill_df158_with_dates(df, start_date=train_dates[0], end_date= valid_dates[1]):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d', errors='coerce')

    def fill_group(g):
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        g = g.set_index('timestamp').sort_index()
        g = g.reindex(full_dates)
        g['item_id'] = g['item_id'].ffill().bfill()
        g = g.reset_index().rename(columns={'index': 'timestamp'})
        return g

    df_filled = df.groupby('item_id', group_keys=False).apply(fill_group)
    return df_filled

df158_filled = fill_df158_with_dates(df158)

# =============== 9️⃣ 转换日期格式以便合并 ===============
df158_filled['timestamp'] = pd.to_datetime(df158_filled['timestamp'])
df_financial_filled['item_id'] = df_financial_filled['item_id'].astype(int)
df158_filled['item_id'] = df158_filled['item_id'].astype(int)
df_financial_filled['timestamp'] = pd.to_datetime(df_financial_filled['timestamp'], format='%Y%m%d', errors='coerce')

# =============== 🔟 合并 df158 与 财务表 ===============
df_merged = pd.merge(
    df158_filled,
    df_financial_filled,
    on=['item_id', 'timestamp'],
    how='left'  # 保留 df158 的完整交易日粒度
)

# =============== ⓫ 可选：删除无效行（如前几列都是 NaN 可过滤） ===============
cols_to_check = df_merged.columns[2:50]
df_merged = df_merged[~df_merged[cols_to_check].isna().all(axis=1)].reset_index(drop=True)

# =============== ⓬ 保存合并结果 ===============
df_merged.to_pickle(project_dir / "temp/qlib_alpha158_ranked_with_stock_finance_info.pkl")
df_merged.to_csv(project_dir / "temp/stage1/df_merged_with_financial.csv", index=False, encoding='utf-8-sig')

print(f"[✅] 合并完成并保存: {df_merged.shape}")

# %%
