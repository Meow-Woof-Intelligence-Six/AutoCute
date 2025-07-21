# %%
# stage42_train_price_change.py
# 目标：使用AutoGluon，结合自定义模型和HPO，训练一个预测涨跌幅排名的回归模型。
# 假设：stage30已成功运行，生成了train.pkl, valid.pkl, test.pkl

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
# valid_data_full = pd.concat([train_data_full, valid_data_full])
test_data_full = pd.concat([train_data_full, valid_data_full, test_data_full])
# print(f"总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data_full)}")

# %%

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

validation_set_for_autogluon = TimeSeriesDataFrame.from_data_frame(
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
prediction_length = 3
custom_lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 56, 84]  # kaggle某个人的方案
from autogluon.timeseries.utils.datetime import (
    get_lags_for_frequency,
    get_seasonality,
    get_time_features_for_frequency,
)

# custom_lags = None # autogluon 自动从 'D'这个频率推断
# custom_lags = get_lags_for_frequency("D")  # autogluon 自动从 'D'这个频率推断
# custom_lags
print("custom_lags: ", custom_lags)

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# model_save_path = project_dir / "model/stage4/44_price_agts_best_quality"
model_save_path = project_dir / "model/stage4/44_price_agts_best_quality-fixval-addfin"
model_save_path.mkdir(parents=True, exist_ok=True)
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=TS_TARGET_LABEL,
    known_covariates_names=known_covariates_names,
    eval_metric=TS_METRIC,  # median 对于排名更加重要
    verbosity=4,
    freq="D",
    path=model_save_path,  # 保存模型的路径
)
# 只有Tempoarl Transformer一个模型支持 past。不过反正我们没有past。

# %%
from autogluon.timeseries.models.presets import get_default_hps
from autogluon.timeseries.configs.presets_configs import TIMESERIES_PRESETS_CONFIGS
from auto_config import pretrained_dir


final_hyperparameters = (
    {
        # 深度学习
        "TiDE": {
            "encoder_hidden_dim": 256,
            "decoder_hidden_dim": 256,
            "temporal_hidden_dim": 64,
            "num_batches_per_epoch": 100,
            "lr": 1e-3,
        },
        "DeepAR": {},
        "Chronos": [
            {
                "ag_args": {"name_suffix": "ZeroShot"},
                "model_path": (
                    pretrained_dir / "autogluon/chronos-bolt-base"
                ).as_posix(),
            },
            {
                "ag_args": {"name_suffix": "FineTuned"},
                "model_path": (
                    pretrained_dir / "autogluon/chronos-bolt-small"
                ).as_posix(),
                "fine_tune": True,
                "target_scaler": "standard",
                "covariate_regressor": {
                    "model_name": "CAT",
                    "model_hyperparameters": {"iterations": 1_0000},
                },
                # "model_hyperparameters": {"iterations": 1_000}},
            },
        ],
        "TemporalFusionTransformer": {},
        # 表格学习
        "DirectTabular": {
            "tabular_hyperparameters": {"GBM": {}},
            "max_num_items": 20_000,
            "max_num_samples": 1_000_000,
            "lags": custom_lags,
            "tabular_fit_kwargs": dict(
                # num_gpus=2,
                # fit_strategy='parallel',
                # time_limit=3600 * 4,
                # presets="best_quality",
                # use_bag_holdout=True,
                #  best_quality={‘auto_stack’: True, ‘dynamic_stacking’: ‘auto’, ‘hyperparameters’: ‘zeroshot’}
            ),
            # "target_scaler": None,  # 第一名的方案没有scaling
            # "target_scaler": "min_max",  # 因为要tweedie
            # 'target_scaler': 'mean_abs' # 因为要稀疏; 注意不是 scaler
        },
        "RecursiveTabular": {
            "tabular_hyperparameters": {"GBM": {}},
            "max_num_items": 20_000,
            "max_num_samples": 1_000_000,
            "lags": custom_lags,
            "tabular_fit_kwargs": dict(
                # num_gpus=2,
                # time_limit=3600 * 4,
                # fit_strategy='parallel',
                # presets="best_quality",
                # use_bag_holdout=True,
            ),
            # "target_scaler": None, # 第一名的方案没有scaling
            # "target_scaler ": "min_max",  # 因为要tweedie
            # 'target_scaler': 'mean_abs' # 因为要稀疏
        },
        # 统计学
        # simple
        "Naive": {},
        "SeasonalNaive": {},  # best quality
        # "Average": {},
        # "SeasonalAverage": {}, # 比较垃圾
        # statistical
        "AutoETS": {},  # best quality
        "AutoARIMA": {},
        "AutoCES": {},
        "Theta": {},
        "DynamicOptimizedTheta": {},  # best quality
        # sparse data
        "NPTS": {},
        "ADIDA": {},
        # "CrostonSBA": {},
        "IMAPA": {},
    },
)

explore_hyperparameters = {
    "TemporalFusionTransformer": [
        {
            "ag_args": {"name_suffix": "Default"},
        },
        {
            "ag_args": {"name_suffix": "FineTuned"},
            
        },
    ]
}
from stage49_infras import auto_ag_priority

predictor.fit(
    # 不是 tuning_data = validation_set_for_autogluon,
    refit_full=True,
    num_val_windows=5,
    train_data=validation_set_for_autogluon,
    random_seed=2002,
    # presets="best_quality",
    # hyperparameters=explore_hyperparameters,
    hyperparameters=auto_ag_priority(final_hyperparameters),
    time_limit=60 * 60 * 8,
    # 前期已知的比较垃圾的或者跑不通的模型
    excluded_model_types=["NPTS", "SeasonalNaive", "PatchTST"],
)
# %%
# /home/ye_canming/repos/novelties/ts/comp/AutoCute/code/src/AutogluonModels/ag-20250719_220122
