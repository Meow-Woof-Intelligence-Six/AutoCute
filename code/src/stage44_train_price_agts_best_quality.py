#%%
# stage42_train_price_change.py
# 目标：使用AutoGluon，结合自定义模型和HPO，训练一个预测涨跌幅排名的回归模型。
# 假设：stage30已成功运行，生成了train.pkl, valid.pkl, test.pkl

import os
# [修复] 解决OpenBLAS多线程导致的段错误 (Segmentation Fault)
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['GOTO_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'

from auto_config import project_dir
os.environ["TABPFN_MODEL_CACHE_DIR"] = (project_dir/"data/pretrained").as_posix()
print(f"设置 TABPFN_MODEL_CACHE_DIR 为: {os.environ['TABPFN_MODEL_CACHE_DIR']}")

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.constants import REGRESSION
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

#%%
# --- 1. 配置与环境准备 ---
print("--- [1/6] 加载配置 ---")

# 定义目标标签
TARGET_LABEL = '收盘_shift' 

# 路径配置
FEATURE_JSON_PATH = project_dir / "temp/stage2/feature_selection_results_vetted.json"
TRAIN_DATA_PATH = project_dir / "temp/stage3/train.pkl"
VALID_DATA_PATH = project_dir / "temp/stage3/valid.pkl"
TEST_DATA_PATH = project_dir / "temp/stage3/test.pkl"
MODEL_OUTPUT_BASE_PATH = project_dir / "models/stage4"
MODEL_OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

# 加载特征选择JSON
with open(FEATURE_JSON_PATH, 'r', encoding='utf-8') as f:
    feature_config = json.load(f)

vetted_features = feature_config.get(TARGET_LABEL, {}).get('final_results', {}).get('vetted_features', [])
categorical_features = feature_config.get('categorical_features_to_keep', [])
features_to_use = vetted_features + categorical_features

if not vetted_features:
    raise ValueError(f"未能从特征选择文件 {FEATURE_JSON_PATH} 中为目标 {TARGET_LABEL} 找到'vetted_features'。请先运行stage2脚本。")

print(f"将使用 {len(features_to_use)} 个特征进行训练。")

#%%
# --- 3. 数据加载与准备 ---
print("\n--- [3/6] 加载预划分的数据集 ---")
train_df = pd.read_pickle(TRAIN_DATA_PATH)
valid_df = pd.read_pickle(VALID_DATA_PATH)
test_df = pd.read_pickle(TEST_DATA_PATH)

# 数据类型修复
print("正在检查并修复数据类型以兼容AutoGluon...")
for df_ in [train_df, valid_df, test_df]:
    for col in df_.columns:
        if str(df_[col].dtype) == 'Int64':
            df_[col] = df_[col].astype('float32')

#%%
#%%
# TODO 静态变量退化
from auto_config import project_dir
import pandas as pd
all_stock_info_df = pd.read_pickle(project_dir / "temp/all_stock_info_df_cleaned.pkl").reset_index()
static_feature_cols = all_stock_info_df.columns.tolist()
static_feature_cols = ["item_id"]+[feat for feat in static_feature_cols if feat in features_to_use]
all_stock_info_df  = all_stock_info_df[static_feature_cols]
all_stock_info_df
#%%
# 退化为 没有 lag 的数据
TS_TARGET_LABEL = '收盘'  # 真实的目标标签
TS_METRIC = "MASE"
# 选择所需的特征和标签
final_cols = ["item_id", "timestamp"]+[feat for feat in features_to_use if TS_TARGET_LABEL not in feat and feat not in static_feature_cols] + [TS_TARGET_LABEL]
final_cols
#%%
train_data_full = train_df[final_cols].dropna(subset=[TS_TARGET_LABEL])
valid_data_full = valid_df[final_cols].dropna(subset=[TS_TARGET_LABEL])
test_data_full = test_df[final_cols] 

print(f"总训练数据: {len(train_data_full)}, 总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data_full)}")
#%%
# 对于时序来说，三个应该叠起来
valid_data_full = pd.concat([train_data_full, valid_data_full])
# test_data_full = pd.concat([train_data_full, valid_data_full, test_data_full])
# print(f"总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data_full)}")

#%%

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
# train_data_ts = TimeSeriesDataFrame.from_data_frame(
#     train_data_full,
#     id_column='item_id',  # 假设有一个item_id列
#     timestamp_column='timestamp',  # 假设有一个timestamp列
# )
validation_set_for_autogluon = TimeSeriesDataFrame.from_data_frame(
    valid_data_full,
    id_column='item_id',  # 假设有一个item_id列
    timestamp_column='timestamp',  # 假设有一个timestamp列
    static_features_df=all_stock_info_df,  # 添加静态特征
)
# test_data_for_autogluon = TimeSeriesDataFrame.from_data_frame(
#     test_data_full,
#     id_column='item_id',  # 假设有一个item_id列
#     timestamp_column='timestamp',  # 假设有一个timestamp列
#     static_features_df=all_stock_info_df,  # 添加静态特征
# )

#%%
known_covariates_names = [col for col in final_cols if "timestamp" in col and col!= "timestamp"]
known_covariates_names
#%%
prediction_length = 3
# custom_lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 56, 84] # kaggle某个人的方案
from autogluon.timeseries.utils.datetime import (
    get_lags_for_frequency,
    get_seasonality,
    get_time_features_for_frequency,
)
# custom_lags = None # autogluon 自动从 'D'这个频率推断
custom_lags = get_lags_for_frequency('D') # autogluon 自动从 'D'这个频率推断
# custom_lags
print("ag preq: ", custom_lags)

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=TS_TARGET_LABEL,
    known_covariates_names=known_covariates_names,
    eval_metric=TS_METRIC,  # median 对于排名更加重要
    verbosity=4,
    freq="D",
)
# 只有Tempoarl Transformer一个模型支持 past。不过反正我们没有past。

# %%
predictor.fit(
    # 不是 tuning_data = validation_set_for_autogluon,
    refit_full=True,
    num_val_windows=5,
    train_data=validation_set_for_autogluon,
    random_seed=2002,
    presets="best_quality",
    time_limit=60 * 60 * 8,
    # 前期已知的比较垃圾的或者跑不通的模型
    excluded_model_types=["NPTS", "SeasonalNaive", "PatchTST"
                          ],
)
# %%
