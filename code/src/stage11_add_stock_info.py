# ycm
#%%
from auto_config import project_dir
import pandas as pd
import joblib
all_stock_info_df = joblib.load(project_dir / "data/additional/all_stock_info_df.joblib")
# 将 上市时间 列转换为 datetime 类型
all_stock_info_df['上市时间'] = pd.to_datetime(all_stock_info_df['上市时间'], format='%Y%m%d')
all_stock_info_df = all_stock_info_df.convert_dtypes()
all_stock_info_df

#%%
should_not_use = [
    "最新", # 泄露未来信息
    "总市值", # 当前股价决定，泄露了当前股价
    "流通市值",
    "prompt"
]
all_stock_info_df = all_stock_info_df.rename(columns={
    "股票代码": "item_id",
    "股票简称": "Name",
    "总股本": "TotalShares",
    "流通股": "CirculatingShares",
    "行业": "Industry",
    "上市时间": "ListingDate",
    }
).drop(columns=should_not_use)
all_stock_info_df['item_id'] = all_stock_info_df['item_id'].astype(int)
# 
all_stock_info_df = all_stock_info_df.set_index('item_id')
all_stock_info_df
#%%
all_stock_info_df.to_pickle(project_dir / "temp/all_stock_info_df_cleaned.pkl")

#%%





# %%
df158 = pd.read_pickle(project_dir / "temp/qlib_alpha158_ranked.pkl")
df360 = pd.read_pickle(project_dir / "temp/qlib_alpha360_ranked.pkl")
# %%


df158 = df158.reset_index()      # 得到 'timestamp'、'item_id' 两列
df158 = df158.merge(all_stock_info_df,
                    left_on='item_id',          # df158 里的股票代码列名
                    right_index=True,        # all_stock_info_df 的索引就是股票代码
                    how='left')              # 以 df158 为主表
df158
#%%
# 上市了多久更有意义
import pandas as pd
from feature_engine.datetime import DatetimeSubtraction
dtf = DatetimeSubtraction(
    variables="timestamp",
    reference="ListingDate",
    output_unit="Y")

df158 = dtf.fit_transform(df158)
df158 = df158.drop(columns=["ListingDate"])  # 删除上市时间列
df158
#%%
df158.to_pickle(project_dir / "temp/qlib_alpha158_ranked_with_stock_info.pkl")

# %%
