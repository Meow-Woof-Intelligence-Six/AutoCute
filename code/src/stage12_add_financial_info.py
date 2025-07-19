#%%
import pandas as pd
from pathlib import Path
from auto_config import project_dir
from auto_config import qlib_dir

# 读入财务数据
#Assets_Liabilities_csv_path = project_dir/"data/finance/Assets_Liabilities.csv" 
#Financial_csv_path = project_dir/"data/finance/Financial.csv" 
#profit_csv_path = project_dir/"data/finance/profit.csv" 
# 1. 读沪深300成分股（空格分隔，只要空格前部分，再去零）
hs300_df = pd.read_csv(qlib_dir /
    "instruments/all.txt",
    sep='\t',          # 文件里是 tab 分隔
    header=None,
    names=['code', 'start_date', 'end_date']
)
hs300 = hs300_df['code'].astype(str).str.lstrip('0').tolist()
hs300
#%%
# 2. 读三张财务表
Assets_Liabilities_df = pd.read_csv(project_dir / "data/finance/Assets_Liabilities.csv")
Financial_df          = pd.read_csv(project_dir / "data/finance/Financial.csv")
profit_df             = pd.read_csv(project_dir / "data/finance/profit.csv")
profit_df  


# 读入之前汇总的pkl数据
df158 = pd.read_pickle(project_dir / "temp/qlib_alpha158_ranked_with_stock_info.pkl")

#%%
# 3) 统一财务表股票代码格式（去 0 前缀）
for df in (Assets_Liabilities_df, Financial_df, profit_df):
    df['股票代码'] = df['股票代码'].astype(str).str.lstrip('0')

# 4) 过滤
Assets_Liabilities1 = Assets_Liabilities_df[Assets_Liabilities_df['股票代码'].isin(hs300)].reset_index(drop=True)
Financial1          = Financial_df[Financial_df['股票代码'].isin(hs300)].reset_index(drop=True)
profit1             = profit_df[profit_df['股票代码'].isin(hs300)].reset_index(drop=True)
# %%
Assets_Liabilities1
# %%
## 排序
for df_name in ['Assets_Liabilities1', 'Financial1', 'profit1']:
    df = globals()[df_name]
    
    # 删除“序号”列（若存在）
    if '序号' in df.columns:
        df = df.drop(columns=['序号'])
    
    # 重排列顺序
    if '股票代码' in df.columns and 'date' in df.columns:
        other_cols = [col for col in df.columns if col not in ['股票代码', 'date']]
        df = df[['股票代码', 'date'] + other_cols]
    else:
        print(f"Warning: {df_name} missing '股票代码' or 'date' columns.")
    
    # 排序
    df = df.sort_values(by=['股票代码', 'date']).reset_index(drop=True)
    
    # 更新回全局变量
    globals()[df_name] = df
# %%
Assets_Liabilities1
Financial1
profit1
# %%
## 列名转换
for df_name in ['Assets_Liabilities1', 'Financial1', 'profit1']:
    df = globals()[df_name]
    
    # 重命名列
    df = df.rename(columns={'股票代码': 'item_id', 'date': 'timestamp'})
    
    # 更新回全局变量
    globals()[df_name] = df

# %%
# 3. 合并财务三表
df_financial = (Assets_Liabilities1
                .merge(Financial1, on=['item_id', 'timestamp'], how='outer', suffixes=('_al', '_fin'))
                .merge(profit1,   on=['item_id', 'timestamp'], how='outer'))
df_financial
#%%
import pandas as pd

def fill_financial_complete_date(df, start_date='2015-04-20', end_date='2025-04-25'):
    # 1. 删除指定列（存在则删）
    cols_to_drop = ['股票简称_fin', '最新公告日期', '股票简称_al', '公告日期_y']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 2. 股票简称和所处行业按 item_id 填充（组内前后向填充）
    for col in ['股票简称', '所处行业']:
        if col in df.columns:
            df[col] = df.groupby('item_id')[col].transform(lambda x: x.ffill().bfill())
    
    # 3. 转 timestamp 为 datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d', errors='coerce')
    
    # 4. 按 item_id 分组，补齐完整日期区间并填充缺失值
    def fill_group(g):
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        g = g.set_index('timestamp').sort_index()
        g = g.reindex(full_dates)
        
        # 填充item_id、股票简称、所处行业列（先前向再后向）
        g['item_id'] = g['item_id'].fillna(method='ffill').fillna(method='bfill')
        for col in ['股票简称', '所处行业']:
            if col in g.columns:
                g[col] = g[col].fillna(method='ffill').fillna(method='bfill')
        
        # 其他数值列
        fill_cols = [c for c in g.columns if c not in ['item_id', '股票简称', '所处行业']]
        
        known_dates = g.dropna(subset=fill_cols, how='all').index
        missing_dates = g[g[fill_cols].isnull().all(axis=1)].index
        
        for missing_date in missing_dates:
            diffs = known_dates - missing_date
            left_dates = known_dates[diffs < pd.Timedelta(0)]
            right_dates = known_dates[diffs > pd.Timedelta(0)]
            
            if len(left_dates) == 0 and len(right_dates) > 0:
                nearest_date = right_dates.min()
            elif len(right_dates) == 0 and len(left_dates) > 0:
                nearest_date = left_dates.max()
            else:
                left_diff = missing_date - left_dates.max()
                right_diff = right_dates.min() - missing_date
                nearest_date = left_dates.max() if left_diff <= right_diff else right_dates.min()
            
            g.loc[missing_date, fill_cols] = g.loc[nearest_date, fill_cols]
            # 确保非空关键列
            for col in ['item_id', '股票简称', '所处行业']:
                if col in g.columns:
                    g.loc[missing_date, col] = g.loc[nearest_date, col]
        
        g = g.reset_index().rename(columns={'index': 'timestamp'})
        # 转回字符串格式方便后续使用
        g['timestamp'] = g['timestamp'].dt.strftime('%Y%m%d')
        return g
    
    df_filled = df.groupby('item_id', group_keys=False).apply(fill_group)
    return df_filled

# 使用示例：
df_financial_filled = fill_financial_complete_date(df_financial)

# %%
df_financial_filled['timestamp'] = pd.to_datetime(df_financial_filled['timestamp'], format='%Y%m%d', errors='coerce')
print(df_financial_filled['timestamp'].dtype)
print(df_financial_filled['timestamp'].head())

df_financial_filled
# %%
import pandas as pd

def fill_df158_with_dates(df, start_date='2015-04-20', end_date='2025-04-25'):
    # 确保 timestamp 是 datetime 类型
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d', errors='coerce')
    
    def fill_group(g):
        # 生成完整日期索引
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        g = g.set_index('timestamp').sort_index()
        
        # 重新索引，补齐所有日期，缺失的行自动生成，内容为空
        g = g.reindex(full_dates)
        
        # 填充 item_id（因为新行只有 timestamp，item_id 要填上）
        g['item_id'] = g['item_id'].fillna(method='ffill').fillna(method='bfill')
        
        # 其他列保持缺失，不填充，保持 NaN
        
        g = g.reset_index().rename(columns={'index': 'timestamp'})
        return g
    
    # 按 item_id 分组处理
    df_filled = df.groupby('item_id', group_keys=False).apply(fill_group)
    
    # timestamp 保持 datetime64[ns]，方便后续处理
    return df_filled

# 用法示例：
df158_filled = fill_df158_with_dates(df158)

#%%
df158_filled
# print(df158_filled['timestamp'].dtype)
#%%
# df158_filled['item_id'] = df158_filled['item_id'].astype(str)
# df_financial_filled['item_id'] = df_financial_filled['item_id'].astype(str)
# 去除字符串中的 .0（只去掉末尾 .0 的情况，防止误删小数部分）
df158_filled['item_id'] = df158_filled['item_id'].str.replace(r'\.0$', '', regex=True)

df_merged1 = pd.merge(
    df158_filled,
    df_financial_filled,
    on=['item_id', 'timestamp'],
    how='outer',  # 根据需要改成 'outer inner'、'left' 或 'right'
    suffixes=('_df158', '_df_financial')
)

df_merged1


#%%
df_merged1.to_pickle(project_dir / "temp/qlib_alpha158_ranked_with_stock_finance_info.pkl")










# %%
