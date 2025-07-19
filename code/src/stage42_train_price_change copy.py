#%%
# stage42_train_price_change.py
# 目标：使用AutoGluon，结合自定义模型和HPO，训练一个预测涨跌幅排名的回归模型。
# 假设：stage30已成功运行，生成了train.pkl, valid.pkl, test.pkl

import os
# [修复] 解决OpenBLAS多线程导致的段错误 (Segmentation Fault)
# 当CPU核心数过多时，AutoGluon的某些底层库（如NumPy/SciPy使用的OpenBLAS）
# 可能会因线程数超出预编译限制而崩溃。我们在此将线程数限制在一个安全值。
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['GOTO_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.models import AbstractModel
from autogluon.core.constants import REGRESSION
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

#%%
# --- 1. 配置与环境准备 ---
print("--- [1/5] 加载配置 ---")

# 定义目标标签
# 我们假设'涨跌幅排名_shift'是已经正态分布化的回归目标
TARGET_LABEL = '涨跌_shift' 

# 相信用户的专业性，假设路径存在
from auto_config import project_dir
FEATURE_JSON_PATH = project_dir / "temp/stage2/feature_selection_results_vetted.json"
# [修改] 从stage3加载预先划分好的数据
TRAIN_DATA_PATH = project_dir / "temp/stage3/train.pkl"
VALID_DATA_PATH = project_dir / "temp/stage3/valid.pkl"
TEST_DATA_PATH = project_dir / "temp/stage3/test.pkl"

MODEL_OUTPUT_BASE_PATH = project_dir / "models/stage4"
MODEL_OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

# 加载特征选择JSON
with open(FEATURE_JSON_PATH, 'r', encoding='utf-8') as f:
    feature_config = json.load(f)

# 提取我们需要的特征列表
vetted_features = feature_config.get(TARGET_LABEL, {}).get('final_results', {}).get('vetted_features', [])
categorical_features = feature_config.get('categorical_features_to_keep', [])
features_to_use = vetted_features + categorical_features

if not vetted_features:
    raise ValueError(f"未能从特征选择文件 {FEATURE_JSON_PATH} 中为目标 {TARGET_LABEL} 找到'vetted_features'。请先运行stage2脚本。")

print(f"目标标签: {TARGET_LABEL}")
print(f"将使用 {len(features_to_use)} 个特征进行训练。")
print(f"其中类别型特征: {len(categorical_features)} 个。")


#%%
# --- 2. 自定义模型实现 (Custom Models) ---
# 用户的秘诀：实现AutoGluon默认没有的，或者有但我们想深度定制的模型
print("\n--- [2/5] 定义自定义模型 ---")
# 假设自定义模型已在 custom_ag 目录中定义好
from custom_ag.ag_svm import AgSVMModel
from custom_ag.ag_nb import IntelligentNaiveBayesModel
from custom_ag.ag_tabpfn import TabPFNModel
# from autogluon.tabular.models.lr.lr_rapids_model import LinearRapidsModel
from autogluon.tabular.models.lr.lr_model import LinearModel

# from autogluon.tabular.configs.hyperparameter_configs import 

#%%
# --- 3. 数据加载与准备 ---
print("\n--- [3/5] 加载预划分的数据集 ---")
train_df = pd.read_pickle(TRAIN_DATA_PATH)
valid_df = pd.read_pickle(VALID_DATA_PATH)
test_df = pd.read_pickle(TEST_DATA_PATH)

# [修复] 解决 "Cannot interpret 'Int64Dtype()'" 错误
# AutoGluon的某些底层库（如Numpy）不识别Pandas的nullable Int64类型。
# 我们在这里将所有这种类型的列安全地转换为float32，这既能保留NA值，又能被AutoGluon正确处理。
print("正在检查并修复数据类型以兼容AutoGluon...")
for df_ in [train_df, valid_df, test_df]:
    for col in df_.columns:
        if str(df_[col].dtype) == 'Int64':
            print(f"  - 发现不兼容类型 Int64 in column '{col}'. 正在转换为 float32...")
            df_[col] = df_[col].astype('float32')

# 选择所需的特征和标签
final_cols = features_to_use + [TARGET_LABEL]
train_data = train_df[final_cols].dropna(subset=[TARGET_LABEL])
valid_data = valid_df[final_cols].dropna(subset=[TARGET_LABEL])
# 测试集可能没有标签，所以只选择特征
test_data = test_df[features_to_use] 

# 转换为AutoGluon的Dataset格式
train_ag = TabularDataset(train_data)
valid_ag = TabularDataset(valid_data)
test_ag = TabularDataset(test_data)

print(f"训练集大小: {len(train_ag)}, 验证集大小: {len(valid_ag)}, 测试集大小: {len(test_ag)}")


#%%
# --- 4. 训练流程 ---
print("\n--- [4/5] 开始三阶段训练流程 ---")

# [修正] 更改评估指标为 'spearmanr'，更适合评估排名预测任务
EVAL_METRIC = 'spearmanr'

# === 阶段一: 广泛探索 (Broad Exploration) ===
print("\n===== [阶段一] 开始广泛探索模型... =====")
from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
initial_models = {
    # **hyperparameter_config_dict['zeroshot_hpo_hybrid'],  # 使用AutoGluon的零-shot HPO配置
    LinearModel: {}, 
    AgSVMModel: {},  # 自定义SVM模型
}
# 制造所有模型类型
for k, v in hyperparameter_config_dict.items():
    # if k.startswith('zeroshot') or k.startswith('hpo'):
        # continue  # 跳过零-shot和HPO配置
    for kk, vv in v.items():
        initial_models[kk] = {}
        


predictor_explore = TabularPredictor(label=TARGET_LABEL, problem_type=REGRESSION, eval_metric=EVAL_METRIC, path=MODEL_OUTPUT_BASE_PATH / "phase1_explore")
predictor_explore.fit(train_data=train_ag, tuning_data=valid_ag, hyperparameters=initial_models, time_limit=1800)
print("\n--- [阶段一] 探索完成 ---")
leaderboard_explore = predictor_explore.leaderboard(valid_ag)
print(leaderboard_explore)
best_model_name = leaderboard_explore.iloc[0]['model']
print(f"\n[阶段一] 结论: 表现最好的模型是: {best_model_name}")

#%%
# === 阶段二: 深度优化 (Hyperparameter Optimization) ===
print("\n===== [阶段二] 开始对顶尖模型进行HPO... =====")
# 我们可以选择对表现最好的一个或几个模型进行HPO
# 这里我们以最佳模型为例
hpo_models = {best_model_name: {}}
predictor_hpo = TabularPredictor(label=TARGET_LABEL, problem_type=REGRESSION, eval_metric=EVAL_METRIC, path=MODEL_OUTPUT_BASE_PATH / "phase2_hpo")
predictor_hpo.fit(train_data=train_ag, tuning_data=valid_ag, hyperparameters=hpo_models, hyperparameter_tune_kwargs={'num_trials': 10, 'searcher': 'auto', 'scheduler': 'auto'}, time_limit=3600)
print("\n--- [阶段二] HPO完成 ---")
leaderboard_hpo = predictor_hpo.leaderboard(valid_ag)
print(leaderboard_hpo)


# === 阶段三: 最终集成 (Final Ensemble) ===
print("\n===== [阶段三] 开始集成所有优化后的模型... =====")
# AutoGluon的fit会自动集成所有训练过的模型，我们可以通过fit_weighted_ensemble来加强
predictor_final = TabularPredictor(label=TARGET_LABEL, problem_type=REGRESSION, eval_metric=EVAL_METRIC, path=MODEL_OUTPUT_BASE_PATH / "phase3_final_ensemble")
predictor_final.fit(train_data=train_ag, tuning_data=valid_ag, time_limit=600, fit_weighted_ensemble=True)
print("\n--- [阶段三] 最终集成完成 ---")
leaderboard_final = predictor_final.leaderboard(valid_ag)
print(leaderboard_final)


#%%
# --- 5. 最终评估与预测 ---
print("\n--- [5/5] 最终评估与生成预测文件 ---")
final_performance = predictor_final.evaluate(valid_ag)
print(f"最终集成模型在【验证集】上的表现: {final_performance}")

# 对测试集进行预测
predictions = predictor_final.predict(test_ag)
submission = pd.DataFrame({'item_id': test_df['item_id'], 'timestamp': test_df['timestamp'], 'prediction': predictions})

# 保存提交文件
submission_path = MODEL_OUTPUT_BASE_PATH / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\n预测结果已保存到: {submission_path}")
print(submission.head())

print("\n===== 训练流程全部结束 =====")
