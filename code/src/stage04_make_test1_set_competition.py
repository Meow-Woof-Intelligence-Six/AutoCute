#%%
from auto_config import project_dir, valid_csv_path, today, memory, test1_dates, test_dates

import pandas as pd
df = pd.read_csv(valid_csv_path)
df = df[["股票代码", "日期", "涨跌幅"]]
df = df.rename(
        columns={"股票代码": "item_id", "日期": "timestamp"}
)
df = df[df["timestamp"] == test_dates[0]]
df = df.sort_values(by="涨跌幅", ascending=False)
df
# %%
top_10_item_ids = df["item_id"].head(10).astype(int).tolist()
# 提取涨幅最小的后10个item_id
bottom_10_item_ids = df["item_id"].tail(10).astype(int).tolist()
final_result_df = pd.DataFrame(
        {"涨幅最大股票代码": top_10_item_ids, "涨幅最小股票代码": bottom_10_item_ids}
    )
final_result_df
# %%
output_path = project_dir / "temp/ref/test_set1_competition.csv"
final_result_df.to_csv(output_path, index=False)
# %%
