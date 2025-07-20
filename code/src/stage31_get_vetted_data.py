#%%
import os
# [修复] 解决OpenBLAS多线程导致的段错误 (Segmentation Fault)
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['GOTO_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'

from auto_config import project_dir, pretrained_dir
os.environ["TABPFN_MODEL_CACHE_DIR"] = pretrained_dir.as_posix()
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
# FEATURE_JSON_PATH = project_dir / "temp/stage2/feature_selection_results_vetted.json"
FEATURE_JSON_PATH = project_dir / "temp/stage2/feature_selection_finance_results_vetted.json"
TRAIN_DATA_PATH = project_dir / "temp/stage3/train.pkl"
VALID_DATA_PATH = project_dir / "temp/stage3/valid.pkl"
TEST_DATA_PATH = project_dir / "temp/stage3/test.pkl"
TEST1_DATA_PATH = project_dir / "temp/stage3/test1.pkl"

#%%
from auto_config import memory, today

@memory.cache
def get_train_valid_test_data(
    TARGET_LABEL = '涨跌幅排名_shift', 
    today =  today,
):
    # 加载特征选择JSON
    with open(FEATURE_JSON_PATH, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)
    vetted_features = feature_config.get(TARGET_LABEL, {}).get('final_results', {}).get('vetted_features', [])
    categorical_features = feature_config.get('categorical_features_to_keep', [])
    features_to_use = vetted_features + categorical_features
    print(f"将使用 {len(features_to_use)} 个特征进行训练。")

    train_df = pd.read_pickle(TRAIN_DATA_PATH)
    valid_df = pd.read_pickle(VALID_DATA_PATH)
    test_df = pd.read_pickle(TEST_DATA_PATH)
    test1_df = pd.read_pickle(TEST1_DATA_PATH)

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
    test1_data = test1_df[final_cols] 
    print(f"总训练数据: {len(train_data_full)}, 总验证数据: {len(valid_data_full)}, 总测试数据: {len(test_data)}, 总测试1数据: {len(test1_data)}")
    return train_data_full, valid_data_full, test_data, test1_data, vetted_features, features_to_use, final_cols, test_df, test1_df

from custom_ag.ag_svm import AgSVMModel
from custom_ag.ag_nb import IntelligentNaiveBayesModel
from custom_ag.ag_tabpfn import TabPFNV2Model
from autogluon.tabular.models.lr.lr_model import LinearModel
from autogluon.tabular.models.tabpfnmix.tabpfnmix_model import TabPFNMixModel

from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
autogluon_models = {}
for k, v in hyperparameter_config_dict.items():
    for kk, vv in v.items():
        if kk!="AG_AUTOMM":
            autogluon_models[kk] = {}
# autogluon_models

MODEL_OUTPUT_BASE_PATH = project_dir / "model/stage4"
MODEL_OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)
#%%


