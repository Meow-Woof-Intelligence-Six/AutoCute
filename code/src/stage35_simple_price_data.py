# %%
# from autogluon.timeseries.models.local.statsforecast
from auto_config import project_dir, valid_csv_path, today, memory
#%%
@memory.cache
def get_simple_data(TARGET_LABEL = "收盘", today=today):
    import pandas as pd
    df = pd.read_csv(valid_csv_path)
    # 真实的目标标签
    df = df[["股票代码", "日期", TARGET_LABEL]]
    df = df.rename(
        columns={"股票代码": "item_id", "日期": "timestamp", TARGET_LABEL: TARGET_LABEL}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["item_id"] = df["item_id"].astype(int)
    from autogluon.timeseries import TimeSeriesDataFrame

    data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    # 非交易日的处理逻辑
    data = data.convert_frequency(freq="D").fill_missing_values("interpolate")
    return data

TARGET_LABEL = "收盘"
data = get_simple_data(TARGET_LABEL=TARGET_LABEL, today=today)