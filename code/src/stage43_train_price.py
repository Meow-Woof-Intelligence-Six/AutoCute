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
# RUN_MODE = 'FULL_TRAIN' 
CV_FOLDS = 5

# 定义目标标签
TARGET_LABEL = '收盘_shift' 
EVAL_METRIC = 'symmetric_mean_absolute_percentage_error'

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
    from custom_ag.ag_tabpfn import TabPFNV2Model
    from autogluon.tabular.models.lr.lr_model import LinearModel
    print("自定义模型已加载。")
except ImportError:
    print("未找到自定义模型，将使用AutoGluon默认模型。")
    AgSVMModel, IntelligentNaiveBayesModel, TabPFNV2Model, LinearModel = None, None, None, None

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
from scipy.stats import spearmanr
# --- 特征质量检查: 计算Spearman相关性 ---
print("\n--- 特征质量检查: 计算与目标变量的Spearman相关性 ---")

# 合并训练和验证数据用于特征质量分析
combined_data = train_data_full
X = combined_data[features_to_use]
y = combined_data[TARGET_LABEL]

print(f"正在计算 {len(features_to_use)} 个特征与目标变量 '{TARGET_LABEL}' 的Spearman相关性...")

# # 计算每个特征与目标变量的Spearman相关性
corrs = {feat: spearmanr(X[feat], y, nan_policy='omit') for feat in X.columns[10:30]}
corr_df = pd.DataFrame(corrs, index=['corr', 'p']).T
print("特征与目标变量的Spearman相关性：")
print(corr_df.sort_values(by='corr', ascending=False))


#%%

#%%
# --- 4. 定义训练函数 ---
# 将三阶段训练流程封装成一个函数，方便在CV中调用
def run_training_pipeline(train_data, valid_data, model_path_suffix=""):
    model_path = MODEL_OUTPUT_BASE_PATH / model_path_suffix
    
    # === 阶段一: 广泛探索 ===
    print(f"\n===== [阶段一 @ {model_path_suffix}] 广泛探索... =====")
    from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
    autogluon_models = {}
    for k, v in hyperparameter_config_dict.items():
        # if k.startswith('zeroshot') or k.startswith('hpo'):
            # continue  # 跳过零-shot和HPO配置
        for kk, vv in v.items():
            if kk!="AG_AUTOMM":
                autogluon_models[kk] = {}
    # autogluon_models

    initial_models = {
                    #   AgSVMModel: {},
                      LinearModel: {},
                      TabPFNV2Model: {},
                      
                      **autogluon_models, 
                    #   **hyperparameter_config_dict['zeroshot_hpo_hybrid'], 
        } 
    # fitted_models = ["GBM", "RF", "KNN", "CAT", "XT", "FASTAI", "TABPFNMIX", "XGB", "NN_TORCH",
    #                   "IM_RULEFIT", 
    #                   "IM_FIGS", 
    #                  ]

    final_hyperparameters = {
        "XT": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
            {"min_samples_leaf": 1, "max_leaf_nodes": 15000, "max_features": 0.5, "ag_args": {"name_suffix": "_r19", "priority": 20}},
        ],
        "RF": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
            {"min_samples_leaf": 5, "max_leaf_nodes": 50000, "max_features": 0.5, "ag_args": {"name_suffix": "_r5", "priority": 19}},
        ],
        "CAT": [
            {"depth": 5, "l2_leaf_reg": 4.774992314058497, "learning_rate": 0.038551267822920274, "ag_args": {"name_suffix": "_r16", "priority": 6}},
            {"depth": 4, "l2_leaf_reg": 1.9950125740798321, "learning_rate": 0.028091050379971633, "ag_args": {"name_suffix": "_r42", "priority": 5}},
            {"depth": 6, "l2_leaf_reg": 1.8298803017644376, "learning_rate": 0.017844259810823604, "ag_args": {"name_suffix": "_r93", "priority": 4}},
            {"depth": 7, "l2_leaf_reg": 4.81099604606794, "learning_rate": 0.019085060180573103, "ag_args": {"name_suffix": "_r44", "priority": 3}},
        ],
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
            {
                "extra_trees": False,
                "feature_fraction": 0.7248284762542815,
                "learning_rate": 0.07947286942946127,
                "min_data_in_leaf": 50,
                "num_leaves": 89,
                "ag_args": {"name_suffix": "_r158", "priority": 18},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.7832570544199176,
                "learning_rate": 0.021720607471727896,
                "min_data_in_leaf": 3,
                "num_leaves": 21,
                "ag_args": {"name_suffix": "_r118", "priority": 17},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.7113010892989156,
                "learning_rate": 0.012535427424259274,
                "min_data_in_leaf": 16,
                "num_leaves": 48,
                "ag_args": {"name_suffix": "_r97", "priority": 16},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.45555769907110816,
                "learning_rate": 0.009591347321206594,
                "min_data_in_leaf": 50,
                "num_leaves": 110,
                "ag_args": {"name_suffix": "_r71", "priority": 15},
            },
            {
                "extra_trees": False,
                "feature_fraction": 0.40979710161022476,
                "learning_rate": 0.008708890211023034,
                "min_data_in_leaf": 3,
                "num_leaves": 80,
                "ag_args": {"name_suffix": "_r111", "priority": 14},
            },
        ],
        "XGB": {},
        
    }

    
    predictor_explore = TabularPredictor(label=TARGET_LABEL,
                                          problem_type=REGRESSION,
                                            eval_metric=EVAL_METRIC,
                                              path=(model_path / "phase3_explore").as_posix())
    predictor_explore.fit(train_data=train_data, 
                          tuning_data=valid_data,
                            hyperparameters=initial_models,
                            # hyperparameters=final_hyperparameters,
                            # raise_on_no_models_fitted=True
                        #   num_gpus=*1
                          time_limit=8 * 60 * 60,  # 8小时
                          ) # 缩短时间以加速
    leaderboard_explore = predictor_explore.leaderboard(valid_data)
    print(leaderboard_explore)
    best_model_name = leaderboard_explore.iloc[0]['model']
    print(f"\n[阶段一] 结论: 最佳模型是 {best_model_name}")

    return predictor_explore


#%%
# --- 5. 执行主流程 ---
print("\n--- [5/6] 开始执行主流程 ---")

if RUN_MODE == 'QUICK_CV':
    print("\n===== 运行模式: 快速交叉验证 (QUICK_CV) =====")
    
    # 合并训练集和验证集用于交叉验证
    train_valid_df = pd.concat([train_data_full, valid_data_full]).sort_index()
    
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    fold_scores = []
    
    for i, (train_index, val_index) in enumerate(tscv.split(train_valid_df)):
        print(f"\n\n<<<<< 开始交叉验证折叠 {i+1}/{CV_FOLDS} >>>>>")
        train_fold_df = train_valid_df.iloc[train_index]
        valid_fold_df = train_valid_df.iloc[val_index]
        
        print(f"折叠 {i+1}: 训练集大小={len(train_fold_df)}, 验证集大小={len(valid_fold_df)}")
        
        # 转换为AutoGluon Dataset
        train_fold_ag = TabularDataset(train_fold_df)
        valid_fold_ag = TabularDataset(valid_fold_df)
        
        # 在当前折叠上运行训练流程
        predictor_fold = run_training_pipeline(train_fold_ag, valid_fold_ag, model_path_suffix=f"cv_fold_{i+1}")
        
        # 评估并记录分数
        performance = predictor_fold.evaluate(valid_fold_ag)
        score = performance[EVAL_METRIC]
        fold_scores.append(score)
        print(f">>>>> 折叠 {i+1} 完成, {EVAL_METRIC} 分数: {score:.6f} <<<<<")
        
    print("\n\n===== 交叉验证全部完成 =====")
    print(f"所有折叠的分数: {[round(s, 6) for s in fold_scores]}")
    print(f"平均 {EVAL_METRIC} 分数: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")

elif RUN_MODE == 'FULL_TRAIN':
    print("\n===== 运行模式: 完整训练 (FULL_TRAIN) =====")
    
    train_ag = TabularDataset(train_data_full)
    valid_ag = TabularDataset(valid_data_full)
    
    # 运行完整的、长时间的训练流程
    final_predictor = run_training_pipeline(train_ag, valid_ag, model_path_suffix="full_train")
    
    # --- 6. 最终评估与预测 ---
    print("\n--- [6/6] 最终评估与生成预测文件 ---")
    final_performance = final_predictor.evaluate(valid_ag)
    print(f"最终集成模型在【验证集】上的表现: {final_performance}")

    # 对测试集进行预测
    test_ag = TabularDataset(test_data)
    predictions = final_predictor.predict(test_ag)
    
    # 确保 test_df 包含 'item_id' 和 'timestamp'
    submission_df = test_df[['item_id', 'timestamp']].copy()
    submission_df['prediction'] = predictions
    
    # 保存提交文件
    submission_path = MODEL_OUTPUT_BASE_PATH / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\n预测结果已保存到: {submission_path}")
    print(submission_df.head())

else:
    raise ValueError(f"未知的运行模式: '{RUN_MODE}'. 请选择 'QUICK_CV' 或 'FULL_TRAIN'.")

print("\n===== 脚本执行结束 =====")


# TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/ye_canming/repos/novelties/ts/comp/AutoCute/models/stage4/full_train/phase3_explore")