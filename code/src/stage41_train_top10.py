# %%
# 定义目标标签
TARGET_LABEL = "龙虎_shift"
EVAL_METRIC = "roc_auc"
print(f"目标标签: {TARGET_LABEL}, 评估指标: {EVAL_METRIC}")
print(f"\n--- [1] 加载配置与数据 ---")
from stage31_get_vetted_data import get_train_valid_test_data

from stage31_get_vetted_data import MODEL_OUTPUT_BASE_PATH

(
    train_data_full,
    valid_data_full,
    test_data,
    test1_data,
    vetted_features,
    features_to_use,
    final_cols,
    test_df,  # 包含 'item_id' 和 'timestamp'
    test1_df,  # 包含 'item_id' 和 'timestamp'
) = get_train_valid_test_data(
    TARGET_LABEL=TARGET_LABEL,
)
train_data_full

print("\n--- [2] 定义自定义模型 ---")
from stage31_get_vetted_data import (
    AgSVMModel,
    IntelligentNaiveBayesModel,
    TabPFNV2Model,
    TabPFNMixModel,
    LinearModel,
    TabularPredictor,
    TabularDataset,
    REGRESSION,
    autogluon_models,
)
import pandas as pd
from auto_config import project_dir

# %%
# --- 4. 定义训练函数 ---

from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict

autogluon_models = {
    "NN_TORCH": {},
    "NN_TORCH": {"num_epochs": 5},
    "XT": [
        {
            "min_samples_leaf": 1,
            "max_leaf_nodes": 15000,
            "max_features": 0.5,
            "ag_args": {"name_suffix": "_r19", "priority": 20},
        }
    ],
    "XT": [
        {
            "criterion": "gini",
            "ag_args": {
                "name_suffix": "Gini",
                "problem_types": ["binary", "multiclass"],
            },
        },
        {
            "criterion": "entropy",
            "ag_args": {
                "name_suffix": "Entr",
                "problem_types": ["binary", "multiclass"],
            },
        },
        {
            "criterion": "squared_error",
            "ag_args": {
                "name_suffix": "MSE",
                "problem_types": ["regression", "quantile"],
            },
        },
        {
            "min_samples_leaf": 1,
            "max_leaf_nodes": 15000,
            "max_features": 0.5,
            "ag_args": {"name_suffix": "_r19", "priority": 20},
        },
    ],
    "XGB": {"n_estimators": 10},
    "XGB": {},
}

# --- 修正后的模型字典 ---
initial_models = {
    LinearModel: {},
    TabPFNV2Model: {},
    **autogluon_models,
}

predictor_explore = TabularPredictor(
    label=TARGET_LABEL,
    # problem_type=REGRESSION,
    eval_metric="roc_auc",
    path=str(MODEL_OUTPUT_BASE_PATH / "top10_classfication_explore"),
)
predictor_explore.fit(
    train_data=train_data_full,
    tuning_data=valid_data_full,
    hyperparameters=initial_models,
    num_gpus=1,
    #   time_limit=600
)  # 缩短时间以加速
# leaderboard_explore = predictor_explore.leaderboard(valid_data_full)
# pred_proba = predictor_explore.predict_proba(test_data)
# feature_importance = predictor_explore.feature_importance(valid_data_full)

# #%%
# with open(project_dir/"temp/stage4/train_top10_results_summary.txt", "w") as f:

#     # 保存 leaderboard_explore
#     f.write("===== Leaderboard Explore =====\n")
#     f.write(leaderboard_explore.to_string(index=False))
#     f.write("\n\n")

#     # 保存 pred_proba
#     f.write("===== Predicted Probabilities =====\n")
#     f.write(pred_proba.to_string())
#     f.write("\n\n")

#     # 保存 feature_importance
#     f.write("===== Feature Importance =====\n")
#     f.write(feature_importance.to_string(index=False))
#     f.write("\n")
# #%%
# pred_proba = predictor_explore.predict_proba(test_data)
# # 1. 取出 test_df 的前两列
# left_cols = test_df.iloc[:, :2]          # 前两列
# # 2. 与 pred_proba 合并
# pred_proba_result = pd.concat([left_cols, pred_proba], axis=1)
# pred_proba_result.to_csv(project_dir / "temp/stage4/top10_proba.csv",
#                         index=False,
#                         encoding='utf-8')
# feature_importance = predictor_explore.feature_importance(valid_data_full)
# feature_importance.to_csv(project_dir / "temp/stage4/top10_feature_importance.csv",
#                           index=True,        # 保留“特征名”作为第一列
#                           header=True,       # 保留列名
#                           encoding='utf-8')


# %% --- 5. 定义预测和保存结果的函数 ---
def predict_and_save_results(predictor, test_data, test_df, output_prefix, save_importance=False):
    # 1. 预测概率
    pred_proba = predictor.predict_proba(test_data)

    # 2. 取出 test_data 的前两列
    left_cols = test_df.iloc[:, :2]  # 前两列

    # 3. 与 pred_proba 合并
    pred_proba_result = pd.concat([left_cols, pred_proba], axis=1)

    # 4. 保存预测概率
    pred_proba_result.to_csv(
        project_dir / f"temp/stage4/{output_prefix}_proba.csv",
        index=False,
        encoding="utf-8",
    )

    # 5. 保存特征重要性
    if save_importance:
        
        feature_importance = predictor.feature_importance(valid_data_full)
        feature_importance.to_csv(
            project_dir / f"temp/stage4/{output_prefix}_feature_importance.csv",
            index=True,  # 保留“特征名”作为第一列
            header=True,  # 保留列名
            encoding="utf-8",
        )


# %% --- 6. 第一次运行：使用 test1_data ---
predict_and_save_results(predictor_explore, test1_data, test1_df, "top10_last10_test1")

# %% --- 7. 第二次运行：使用 test_data ---
predict_and_save_results(predictor_explore, test_data, test_df, "top10_last10_test")
