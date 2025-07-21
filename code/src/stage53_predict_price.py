#%%
TARGET_LABEL = '收盘_shift'
from stage31_get_vetted_data import MODEL_OUTPUT_BASE_PATH
import pandas as pd

predictions = pd.read_csv(MODEL_OUTPUT_BASE_PATH / f"{TARGET_LABEL}_test.csv")
predictions1 = pd.read_csv(MODEL_OUTPUT_BASE_PATH / f"{TARGET_LABEL}_test1.csv")


#%%
from auto_config import project_dir, valid_csv_path, today, memory, test1_dates, test_dates

import pandas as pd
df = pd.read_csv(valid_csv_path)
df = df[["股票代码", "日期", "收盘"]]
df = df.rename(
        columns={"股票代码": "item_id", "日期": "timestamp"}
)
df = df[df["timestamp"] == test_dates[0]]
df1 = df[df["timestamp"] == test1_dates[0]]
df
#%%
predictions.rename(
    columns={"prediction": "mean"}, inplace=True
)
predictions1.rename(
    columns={"prediction": "mean"}, inplace=True
)
#%%

# 调用函数
from auto_config import project_dir
from stage59 import predictions_to_competition_df

res = predictions_to_competition_df(
    predictions = predictions, 
    test_data_for_autogluon = df,
    test_date = test_dates[0],
    date_before_test = test_dates[0],
)
res.to_csv(
    project_dir / f"temp/output/results_{TARGET_LABEL}_test.csv", 
    index=False
)
#%%

res1 = predictions_to_competition_df(
    predictions = predictions1, 
    test_data_for_autogluon = df1,
    test_date = test1_dates[0],
    date_before_test = test1_dates[0],
)

res1.to_csv(
    project_dir / f"temp/output/results_{TARGET_LABEL}_test1.csv", 
    index=False
)

# %%
