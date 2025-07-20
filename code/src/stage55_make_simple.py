#%%
from auto_config import project_dir, valid_csv_path
import pandas as pd

df = pd.read_csv(valid_csv_path)
df
# %%
TARGET_LABEL = "收盘"  # 真实的目标标签
df = df[["股票代码", "日期", TARGET_LABEL]]
df = df.rename(
    columns={"股票代码": "item_id", "日期": "timestamp", TARGET_LABEL: TARGET_LABEL}
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["item_id"] = df["item_id"].astype(int)
# df.dtypes
# df
# %%
from autogluon.timeseries import TimeSeriesDataFrame

data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
)
# 非交易日的处理逻辑
data = data.convert_frequency(freq="D").fill_missing_values("interpolate")
data
# %%
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
model_path = project_dir / "model/stage45_train_price_agts_stats"
predictor = TimeSeriesPredictor.load(model_path)
# %%
predictions = predictor.predict(data)
predictions
#%%
from auto_config import date_before_test, test_date
from stage59 import predictions_to_competition_df
model_mode = "price"
model_name = predictor.path.split("/")[-1]
result_df = predictions_to_competition_df(
    predictions,
    test_date = test_date,
    date_before_test=date_before_test,
    test_data_for_autogluon=data,
    model_mode=model_mode
)
result_df
# %%
result_df.to_csv(
    project_dir / f"output/result-{model_mode}-{model_name}.csv", index=False
)

# %%
