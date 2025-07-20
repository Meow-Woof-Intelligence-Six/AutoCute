# %%
# from autogluon.timeseries.models.local.statsforecast
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
# metric_name = "MASE"
metric_name = "RMSSE"
prediction_length = 3

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    # target="收盘",
    target=TARGET_LABEL,
    eval_metric=metric_name,  # Primary metric for training
    verbosity=4,
    freq="D",
    path=project_dir
    / "model/stage45_train_price_agts_stats",  # Save model to this path
)

predictor.fit(
    refit_full=True,
    num_val_windows=2,
    train_data=data,
    random_seed=2002,
    hyperparameters={
        # simple
        "Naive": {},
        "SeasonalNaive": {},
        "Average": {},
        "SeasonalAverage": {},
        # statistical
        "AutoETS": {},
        "AutoARIMA": {},
        "AutoCES": {},
        "Theta": {},
        "DynamicOptimizedTheta": {},
        # sparse data
        "NPTS": {},
        "ADIDA": {},
        "CrostonSBA": {},
        "IMAPA": {},
    },
)
