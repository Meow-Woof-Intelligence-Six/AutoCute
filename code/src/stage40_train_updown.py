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

# [新增] 运行模式选择: 'QUICK_CV' 或 'FULL_TRAIN'
RUN_MODE = 'QUICK_CV' 
CV_FOLDS = 5

# 定义目标标签
TARGET_LABEL = '涨跌_shift' 
EVAL_METRIC = 'roc_auc'

# 路径配置
# FEATURE_JSON_PATH = project_dir / "temp/stage2/feature_selection_results_vetted.json"
FEATURE_JSON_PATH = project_dir / "temp/stage2/feature_selection_finance_results_vetted.json"
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

print(f"运行模式: {RUN_MODE}")
print(f"目标标签: {TARGET_LABEL}, 评估指标: {EVAL_METRIC}")
print(f"将使用 {len(features_to_use)} 个特征进行训练。")


#%%
# --- 2. 自定义模型实现 (Custom Models) ---
print("\n--- [2/6] 定义自定义模型 ---")
# 假设自定义模型已在 custom_ag 目录中定义好
try:
    from custom_ag.ag_svm import AgSVMModel
    from custom_ag.ag_nb import IntelligentNaiveBayesModel
    from custom_ag.ag_tabpfn import TabPFNModel
    from autogluon.tabular.models.lr.lr_model import LinearModel
    print("自定义模型已加载。")
except ImportError:
    print("未找到自定义模型，将使用AutoGluon默认模型。")
    AgSVMModel, IntelligentNaiveBayesModel, TabPFNModel, LinearModel = None, None, None, None

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

# 选择所需的特征和标签
final_cols = features_to_use + [TARGET_LABEL]
train_data_full = train_df[final_cols].dropna(subset=[TARGET_LABEL])
valid_data_full = valid_df[final_cols].dropna(subset=[TARGET_LABEL])
test_data = test_df[final_cols] 

print(f"总训练数据: {len(train_data_full)}, 总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data)}")
#%%
train_data_full
#%%
# --- 4. 定义训练函数 ---

from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
autogluon_models = {
    "FASTAI": {},
    "FASTAI": [
        {
            "bs": 1024,
            "emb_drop": 0.6167722379778131,
            "epochs": 44,
            "layers": [200, 100, 50],
            "lr": 0.053440377855629266,
            "ps": 0.48477211305443607,
            "ag_args": {"name_suffix": "_r25", "priority": 13},
        },
        {
            "bs": 1024,
            "emb_drop": 0.6046989241462619,
            "epochs": 48,
            "layers": [200, 100, 50],
            "lr": 0.00775309042164966,
            "ps": 0.09244767444160731,
            "ag_args": {"name_suffix": "_r51", "priority": 12},
        },
        {
            "bs": 512,
            "emb_drop": 0.6557225316526186,
            "epochs": 49,
            "layers": [200, 100],
            "lr": 0.023627682025564638,
            "ps": 0.519566584552178,
            "ag_args": {"name_suffix": "_r82", "priority": 11},
        },
        {
            "bs": 2048,
            "emb_drop": 0.4066210919034579,
            "epochs": 43,
            "layers": [400, 200],
            "lr": 0.0029598312717673434,
            "ps": 0.4378695797438974,
            "ag_args": {"name_suffix": "_r121", "priority": 10},
        },
        {
            "bs": 128,
            "emb_drop": 0.44339037504795686,
            "epochs": 31,
            "layers": [400, 200, 100],
            "lr": 0.008615195908919904,
            "ps": 0.19220253419114286,
            "ag_args": {"name_suffix": "_r145", "priority": 9},
        },
        {
            "bs": 128,
            "emb_drop": 0.12106594798980945,
            "epochs": 38,
            "layers": [200, 100, 50],
            "lr": 0.037991970245029975,
            "ps": 0.33120008492595093,
            "ag_args": {"name_suffix": "_r173", "priority": 8},
        },
        {
            "bs": 128,
            "emb_drop": 0.4599138419358,
            "epochs": 47,
            "layers": [200, 100],
            "lr": 0.03888383281136287,
            "ps": 0.28193673177122863,
            "ag_args": {"name_suffix": "_r128", "priority": 7},
        },
    ],

    "XT": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
        {"min_samples_leaf": 1, "max_leaf_nodes": 15000, "max_features": 0.5, "ag_args": {"name_suffix": "_r19", "priority": 20}},
    ],
}
# autogluon_models

# initial_models = {
#                 #   AgSVMModel: {},
#                     LinearModel: {},
#                     TabPFNModel: {},
                    
#                     **autogluon_models, 
#                 #   **hyperparameter_config_dict['zeroshot_hpo_hybrid'], 
#     } 

# --- 修正后的模型字典 ---
initial_models = {
    LinearModel: {},
    TabPFNModel: {},
    **autogluon_models,
}
# 过滤掉 value 为 None 的项
initial_models = {k: v for k, v in initial_models.items() if k is not None}


fitted_models = ["GBM", "RF", "KNN", "CAT", "XT", "FASTAI", "TABPFNMIX", "XGB", "NN_TORCH"]

predictor_explore = TabularPredictor(label=TARGET_LABEL,
                                        #problem_type=REGRESSION,
                                        eval_metric='roc_auc',
                                            path= MODEL_OUTPUT_BASE_PATH / "updown_classfication_explore")
predictor_explore.fit(train_data=train_data_full, 
                        tuning_data=valid_data_full,
                        hyperparameters=initial_models, 
                        num_gpus=1
                    #   time_limit=600
                        ) # 缩短时间以加速
leaderboard_explore = predictor_explore.leaderboard(valid_data_full)
pred_proba = predictor_explore.predict_proba(test_data)
feature_importance = predictor_explore.feature_importance(valid_data_full)

#%%
with open(project_dir/"temp/stage4/train_updown_results_summary.txt", "w") as f:

    # 保存 leaderboard_explore
    f.write("===== Leaderboard Explore =====\n")
    f.write(leaderboard_explore.to_string(index=False))
    f.write("\n\n")

    # 保存 pred_proba
    f.write("===== Predicted Probabilities =====\n")
    f.write(pred_proba.to_string())
    f.write("\n\n")

    # 保存 feature_importance
    f.write("===== Feature Importance =====\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n")
#%%
