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
TARGET_LABEL = '涨跌幅排名_shift' 
EVAL_METRIC = 'spearmanr'

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
corr_df
# # 筛选显著性特征 (p < 0.05)
# significant_features = corr_df[corr_df['p'] < 0.05]
# print(f"在 {len(features_to_use)} 个特征中，有 {len(significant_features)} 个特征与目标变量存在显著相关性 (p < 0.05)")

# # 按相关性绝对值排序，显示前10个最相关的特征
# top_features = corr_df.sort_values('corr', key=abs, ascending=False).head(10)
# print("\n前10个最相关的特征:")
# print(top_features.round(6))

# # 显示相关性分布统计
# print(f"\n相关性统计:")
# print(f"平均绝对相关性: {corr_df['corr'].abs().mean():.6f}")
# print(f"最大绝对相关性: {corr_df['corr'].abs().max():.6f}")
# print(f"显著特征占比: {len(significant_features)/len(features_to_use)*100:.2f}%")

# # 警告: 如果显著特征太少
# if len(significant_features) < len(features_to_use) * 0.1:
#     print("⚠️  警告: 显著相关特征数量较少，模型性能可能受限")

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
                    #   LinearModel: {},
                      TabPFNModel: {},
                      
                      **autogluon_models, 
                    #   **hyperparameter_config_dict['zeroshot_hpo_hybrid'], 
        } 
    fitted_models = ["GBM", "RF", "KNN", "CAT", "XT", "FASTAI", "TABPFNMIX", "XGB", "NN_TORCH",
                      "IM_RULEFIT", 
                      "IM_FIGS", 
                     ]
    # KNN < GBM < RF
    # predictor = TabularPredictor.load("/home/ye_canming/repos/novelties/ts/comp/AutoCute/models/stage4/cv_fold_5/phase1_explore")
    # Fitting model: LinearModel ...
        # 0.17     = Validation score   (spearmanr)
        # 20.81s   = Training   runtime
    # 
    #  0.1569   = Validation score   (spearmanr)
    # Fitting model: CatBoost ...
    #         0.2256   = Validation score   (spearmanr)
    # Fitting model: ExtraTrees ...
    #         0.2476   = Validation score   (spearmanr)
    # Fitting model: NeuralNetFastAI ...
    #         0.2044   = Validation score   (spearmanr)
    # Fitting model: TabPFNMix ...
    # A maximum of 100 features are allowed, but the dataset has 240 features. A subset of 100 are selected using SelectKBest
    #         0.1635   = Validation score   (spearmanr)
    # Fitting model: XGBoost ...
    #         0.2226   = Validation score   (spearmanr)
    # Fitting model: NeuralNetTorch ...
    #         0.2062   = Validation score   (spearmanr)

    for model_name in fitted_models:
        del initial_models[model_name] 
    
    predictor_explore = TabularPredictor(label=TARGET_LABEL,
                                          problem_type=REGRESSION,
                                            eval_metric=EVAL_METRIC,
                                              path=model_path / "phase1_explore")
    predictor_explore.fit(train_data=train_data, 
                          tuning_data=valid_data,
                            hyperparameters=initial_models, 
                            raise_on_no_models_fitted = True
                        #   num_gpus=1
                        #   time_limit=600
                          ) # 缩短时间以加速
    leaderboard_explore = predictor_explore.leaderboard(valid_data)
    print(leaderboard_explore)
    best_model_name = leaderboard_explore.iloc[0]['model']
    print(f"\n[阶段一] 结论: 最佳模型是 {best_model_name}")

    return predictor_explore
    # # === 阶段二: HPO ===
    # print(f"\n===== [阶段二 @ {model_path_suffix}] HPO... =====")
    # predictor_hpo = TabularPredictor(label=TARGET_LABEL, problem_type=REGRESSION, eval_metric=EVAL_METRIC, path=model_path / "phase2_hpo")
    # predictor_hpo.fit(train_data=train_data, tuning_data=valid_data, hyperparameters={best_model_name: {}}, hyperparameter_tune_kwargs={'num_trials': 5, 'searcher': 'auto'}, time_limit=1200) # 减少trial数量

    # # === 阶段三: 集成 ===
    # print(f"\n===== [阶段三 @ {model_path_suffix}] 集成... =====")
    # predictor_final = TabularPredictor(label=TARGET_LABEL, problem_type=REGRESSION, eval_metric=EVAL_METRIC, path=model_path / "phase3_final_ensemble")
    # predictor_final.fit(train_data=train_data, tuning_data=valid_data, time_limit=300, fit_weighted_ensemble=True)
    
    # return predictor_final

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
