#%%
TARGET_LABEL = '涨跌幅排名_shift'
EVAL_METRIC = 'spearmanr'
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


#%%
# --- 4. 定义训练函数 ---
# 将三阶段训练流程封装成一个函数，方便在CV中调用
def run_training_pipeline(train_data, valid_data, model_path_suffix=""):
    model_path = MODEL_OUTPUT_BASE_PATH / model_path_suffix
    
    # === 阶段一: 广泛探索 ===
    print(f"\n===== [阶段一 @ {model_path_suffix}] 广泛探索... =====")
    

    initial_models = {
                    #   AgSVMModel: {},
                    #   LinearModel: {},
                      TabPFNV2Model: {},
                    #   **autogluon_models, 
                    #   **hyperparameter_config_dict['zeroshot_hpo_hybrid'], 
        } 


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
train_ag = TabularDataset(train_data_full)
valid_ag = TabularDataset(valid_data_full)

predictor_explore = run_training_pipeline(train_ag, valid_ag, model_path_suffix="full_train")

def predict_and_save_results(predictor, test_data, test_df, output_prefix, save_importance=False):
    test_ag = TabularDataset(test_data)
    predictions = predictor.predict(test_ag)
    # 确保 test_df 包含 'item_id' 和 'timestamp'
    submission_df = test_df[['item_id', 'timestamp']].copy()
    submission_df['prediction'] = predictions
    # 保存提交文件
    submission_path = MODEL_OUTPUT_BASE_PATH / f"{output_prefix}.csv"
    submission_df.to_csv(submission_path, index=False, 
                         encoding="utf-8"
        )
        
    print(f"\n预测结果已保存到: {submission_path}")
    print(submission_df.head())
    if save_importance:
        feature_importance = predictor.feature_importance(valid_data_full)
        feature_importance.to_csv(
            MODEL_OUTPUT_BASE_PATH / f"{output_prefix}_feature_importance.csv",
            index=True,  # 保留“特征名”作为第一列
            header=True,  # 保留列名
            encoding="utf-8",
        )

predict_and_save_results(predictor_explore, test1_data, test1_df, f"{TARGET_LABEL}_test1")

predict_and_save_results(predictor_explore, test1_data, test1_df, f"{TARGET_LABEL}_test1")


# TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/ye_canming/repos/novelties/ts/comp/AutoCute/models/stage4/full_train/phase3_explore")