# %%
import os


# [修复] 解决OpenBLAS多线程导致的段错误 (Segmentation Fault)
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["GOTO_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"

from auto_config import project_dir

os.environ["TABPFN_MODEL_CACHE_DIR"] = (project_dir / "data/pretrained").as_posix()
print(f"设置 TABPFN_MODEL_CACHE_DIR 为: {os.environ['TABPFN_MODEL_CACHE_DIR']}")

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.constants import REGRESSION
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# %%
# --- 1. 配置与环境准备 ---
print("--- [1/6] 加载配置 ---")

# 定义目标标签
TARGET_LABEL = "收盘_shift"

# 路径配置
FEATURE_JSON_PATH = (
    project_dir / "temp/stage2/feature_selection_finance_results_vetted.json"
)
TRAIN_DATA_PATH = project_dir / "temp/stage3/train.pkl"
VALID_DATA_PATH = project_dir / "temp/stage3/valid.pkl"
TEST_DATA_PATH = project_dir / "temp/stage3/test.pkl"
MODEL_OUTPUT_BASE_PATH = project_dir / "models/stage4"
MODEL_OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

# 加载特征选择JSON
with open(FEATURE_JSON_PATH, "r", encoding="utf-8") as f:
    feature_config = json.load(f)

vetted_features = (
    feature_config.get(TARGET_LABEL, {})
    .get("final_results", {})
    .get("vetted_features", [])
)
categorical_features = feature_config.get("categorical_features_to_keep", [])
features_to_use = vetted_features + categorical_features

if not vetted_features:
    raise ValueError(
        f"未能从特征选择文件 {FEATURE_JSON_PATH} 中为目标 {TARGET_LABEL} 找到'vetted_features'。请先运行stage2脚本。"
    )

print(f"将使用 {len(features_to_use)} 个特征进行训练。")

# %%
# --- 3. 数据加载与准备 ---
print("\n--- [3/6] 加载预划分的数据集 ---")
train_df = pd.read_pickle(TRAIN_DATA_PATH)
valid_df = pd.read_pickle(VALID_DATA_PATH)
test_df = pd.read_pickle(TEST_DATA_PATH)

# 数据类型修复
print("正在检查并修复数据类型以兼容AutoGluon...")
for df_ in [train_df, valid_df, test_df]:
    for col in df_.columns:
        if str(df_[col].dtype) == "Int64":
            df_[col] = df_[col].astype("float32")

# %%
# %%
# TODO 静态变量退化
from auto_config import project_dir
import pandas as pd

all_stock_info_df = pd.read_pickle(
    project_dir / "temp/all_stock_info_df_cleaned.pkl"
).reset_index()
static_feature_cols = all_stock_info_df.columns.tolist()
static_feature_cols = ["item_id"] + [
    feat for feat in static_feature_cols if feat in features_to_use
]
all_stock_info_df = all_stock_info_df[static_feature_cols]
all_stock_info_df
# %%
# 退化为 没有 lag 的数据
TS_TARGET_LABEL = "收盘"  # 真实的目标标签
TS_METRIC = "MASE"
# 选择所需的特征和标签
final_cols = (
    ["item_id", "timestamp"]
    + [
        feat
        for feat in features_to_use
        if TS_TARGET_LABEL not in feat and feat not in static_feature_cols
    ]
    + [TS_TARGET_LABEL]
)
final_cols
# %%
train_data_full = train_df[final_cols].dropna(subset=[TS_TARGET_LABEL])
valid_data_full = valid_df[final_cols].dropna(subset=[TS_TARGET_LABEL])
test_data_full = test_df[final_cols]

print(
    f"总训练数据: {len(train_data_full)}, 总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data_full)}"
)
# %%
# 对于时序来说，三个应该叠起来
valid_data_full = pd.concat([train_data_full, valid_data_full])
test_data_full = pd.concat([train_data_full, valid_data_full, test_data_full])
print(f"总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data_full)}")

# %%

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

train_data_ts = TimeSeriesDataFrame.from_data_frame(
    train_data_full,
    id_column="item_id",  # 假设有一个item_id列
    timestamp_column="timestamp",  # 假设有一个timestamp列
)
validation_set_for_autogluon = TimeSeriesDataFrame.from_data_frame(
    valid_data_full,
    id_column="item_id",  # 假设有一个item_id列
    timestamp_column="timestamp",  # 假设有一个timestamp列
    static_features_df=all_stock_info_df,  # 添加静态特征
)
test_data_for_autogluon = TimeSeriesDataFrame.from_data_frame(
    test_data_full,
    id_column="item_id",  # 假设有一个item_id列
    timestamp_column="timestamp",  # 假设有一个timestamp列
    static_features_df=all_stock_info_df,  # 添加静态特征
)

# %%
known_covariates_names = [
    col for col in final_cols if "timestamp" in col and col != "timestamp"
]
known_covariates_names
# %%
import joblib

time_features_maker_path = project_dir / "temp/stage2/time_features_maker.pkl"

time_features_maker = joblib.load(time_features_maker_path)
date_features = time_features_maker["date_features"]
cyclical_encoder = time_features_maker["cyclical_encoder"]

# %%
from auto_config import project_dir, model_dir

# model_path = project_dir/"code/src/AutogluonModels/ag-20250720_072646"
# model_path = project_dir / "model/stage4/44_price_agts_best_quality"
model_path = project_dir / "model/stage4/44_price_agts_best_quality-fixval-addfin"
from autogluon.timeseries import TimeSeriesPredictor

# Load the predictor from the specified path
predictor = TimeSeriesPredictor.load(model_path)
predictor.leaderboard()
# %%
test_data_for_autogluon1 = test_data_for_autogluon[test_data_for_autogluon.index.get_level_values('timestamp') < test_data_for_autogluon.index.get_level_values('timestamp').max()]

future_data = predictor.make_future_data_frame(test_data_for_autogluon)
future_data

future_data1 = predictor.make_future_data_frame(test_data_for_autogluon1)
future_data1
# %%
df158 = pd.read_pickle(project_dir / "temp/lag158_finance.pkl")
# 取 df 158 最后 future_data 的行数
df158 = df158.tail(len(future_data)).reset_index(drop=True)
# 用 future_data 的数据覆盖 df158 的数据
for col in future_data.columns:
    if col in df158.columns:
        df158[col] = future_data[col]
# df158

df1581 = pd.read_pickle(project_dir / "temp/lag158_finance.pkl")
# 取 df 158 最后 future_data 的行数
df1581 = df1581.tail(len(future_data1)).reset_index(drop=True)
# 用 future_data 的数据覆盖 df158 的数据
for col in future_data1.columns:
    if col in df1581.columns:
        df1581[col] = future_data1[col]

# %%

# 添加时间特征
future_data_real = date_features.transform(df158)
# 添加周期性特征
future_data_real = cyclical_encoder.transform(future_data_real)
future_data_real = future_data_real[
    future_data.columns.to_list() + known_covariates_names
]
# future_data_real

future_data_real1 = date_features.transform(df1581)
# 添加周期性特征
future_data_real1 = cyclical_encoder.transform(future_data_real1)
future_data_real1 = future_data_real1[
    future_data1.columns.to_list() + known_covariates_names
]
# %%
predictions = predictor.predict(
    test_data_for_autogluon, known_covariates=future_data_real
)

predictions1 = predictor.predict(
    test_data_for_autogluon1, known_covariates=future_data_real1
)

#%%
from auto_config import test1_dates, test_dates
import pandas as pd

# %%
predictions = joblib.load(predictions_path)
from stage59 import predictions_to_competition_df


from auto_config import test1_dates, test_dates
from stage59 import predictions_to_competition_df
from datetime import datetime, timedelta
real_test_date = datetime.strftime(datetime.strptime(test_dates[0], '%Y-%m-%d') + timedelta(days=3), '%Y-%m-%d')
real_test_date

model_mode = "price"
result_df = predictions_to_competition_df(
    predictions,
    test_date = real_test_date,
    date_before_test=test_dates[0],
    test_data_for_autogluon=test_data_for_autogluon,
    model_mode=model_mode
)



model_name = predictor.path.split("/")[-1]

result_df.to_csv(
    project_dir / f"output/result-{model_mode}-{model_name}-test.csv", index=False
)

#%%




# %%
