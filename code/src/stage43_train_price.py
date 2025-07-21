# %%
# 定义目标标签
TARGET_LABEL = "收盘_shift"
EVAL_METRIC = "symmetric_mean_absolute_percentage_error"
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
from auto_config import project_dir, pretrained_dir

# %%
# 将三阶段训练流程封装成一个函数，方便在CV中调用
def run_training_pipeline(train_data, valid_data, model_path_suffix=""):
    model_path = MODEL_OUTPUT_BASE_PATH / model_path_suffix
    # === 阶段一: 广泛探索 ===
    print(f"\n===== [阶段一 @ {model_path_suffix}] 广泛探索... =====")
    from autogluon.tabular.configs.hyperparameter_configs import (
        hyperparameter_config_dict,
    )
    initial_models = {
        #   AgSVMModel: {},
        LinearModel: {},
        TabPFNV2Model: {},
        **autogluon_models,
        #   **hyperparameter_config_dict['zeroshot_hpo_hybrid'],
    }
    fitted_models = [
        "GBM",
        "RF",
        "KNN",
        "CAT",
        "XT",
        "FASTAI",
        "TABPFNMIX",
        "XGB",
        "NN_TORCH",
        "IM_RULEFIT",
        "IM_FIGS",
    ]
    #    Fitting model: KNeighbors ... Training model for up to 28786.79s of the 28786.79s of remaining time.
    #         -0.3409  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         2.49s    = Training   runtime
    #         6.04s    = Validation runtime
    # Fitting model: LightGBM ... Training model for up to 28777.91s of the 28777.91s of remaining time.

    #         -0.016   = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         343.15s  = Training   runtime
    # 1.06s    = Validation runtime

    # Fitting model: RandomForest ... Training model for up to 28433.50s of the 28433.50s of remaining time.
    #         -0.0127  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         321.22s  = Training   runtime
    #         0.4s     = Validation runtime
    # Fitting model: CatBoost ... Training model for up to 28111.09s of the 28111.09s of remaining time.
    #         -0.2166  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         7.42s    = Training   runtime
    #         0.07s    = Validation runtime
    # Fitting model: ExtraTrees ... Training model for up to 28103.58s of the 28103.58s of remaining time.
    #         -0.0127  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         60.93s   = Training   runtime
    #         0.32s    = Validation runtime
    # Fitting model: NeuralNetFastAI ... Training model for up to 28041.73s of the 28041.73s of remaining time.
    # Metric symmetric_mean_absolute_percentage_error is not supported by this model - using mean_squared_error instead
    #         Warning: Exception caused NeuralNetFastAI to fail during training... Skipping this model.
    # Fitting model: TabPFNMix ... Training model for up to 28033.04s of the 28033.04s of remaining time.
    # A maximum of 100 features are allowed, but the dataset has 203 features. A subset of 100 are selected using SelectKBest
    #         -0.0476  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         17.75s   = Training   runtime
    #         450.57s  = Validation runtime
    # Fitting model: XGBoost ... Training model for up to 27563.79s of the 27563.79s of remaining time.
    #         -0.0162  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         20.13s   = Training   runtime
    #         1.32s    = Validation runtime
    # Fitting model: NeuralNetTorch ... Training model for up to 27542.34s of the 27542.34s of remaining time.
    #         -0.2203  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         1010.11s         = Training   runtime
    #         1.58s    = Validation runtime
    # Fitting model: LinearModel ... Training model for up to 26530.65s of the 26530.64s of remaining time.
    #         -0.4001  = Validation score   (-symmetric_mean_absolute_percentage_error)

    #     Fitting model: RuleFit ... Training model for up to 26501.47s of the 26501.47s of remaining time.
    #         -0.1925  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         2246.33s         = Training   runtime
    #         1.47s    = Validation runtime
    # Fitting model: Figs ... Training model for up to 24253.66s of the 24253.66s of remaining time.
    #         -0.1598  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         243.16s  = Training   runtime
    #         0.5s     = Validation runtime
    # Fitting model: WeightedEnsemble_L2 ... Training model for up to 2878.68s of the 24009.97s of remaining time.
    #         Ensemble Weights: {'RandomForest': 0.6, 'ExtraTrees': 0.4}
    #         -0.0125  = Validation score   (-symmetric_mean_absolute_percentage_error)
    #         0.14s    = Training   runtime
    #         0.0s     = Validation runtime

    tabpdn_path_dict = {
            "model_path_classifier": str(
                pretrained_dir / "autogluon/tabpfn-mix-1.0-classifier"
            ),  
            "model_path_regressor": str(
                pretrained_dir / "autogluon/tabpfn-mix-1.0-regressor"
            ), 
    }
    final_hyperparameters = {
        TabPFNV2Model: {},
        TabPFNMixModel: [
            {
                "ag_args": {"name_suffix": "ZeroShot"},
                **tabpdn_path_dict,
            }, 
            # {
            #     "ag_args": {"name_suffix": "FineTunedv1"},
            #     **tabpdn_path_dict,
            #     "n_ensembles": 1,
            #     "max_epochs": 30,
            #     "ag.sample_rows_val": 5000,  # Beyond 5k val rows fine-tuning becomes very slow
            #     "ag.max_rows": 50000,  # Beyond 50k rows, the time taken is longer than most users would like (hours), while the model is very weak at this size
            # }, 
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
        "RF": [
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
                "min_samples_leaf": 5,
                "max_leaf_nodes": 50000,
                "max_features": 0.5,
                "ag_args": {"name_suffix": "_r5", "priority": 19},
            },
        ],
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {
                    "name_suffix": "Large",
                    "priority": 0,
                    "hyperparameter_tune_kwargs": None,
                },
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
    fast = {
        "GBM": {}
    }
    from stage49_infras import auto_ag_priority
    predictor_explore = TabularPredictor(
        label=TARGET_LABEL,
        problem_type=REGRESSION,
        eval_metric=EVAL_METRIC,
        path=(model_path / "phase4_train").as_posix(),
    )
    #   path=(model_path / "phase3_explore").as_posix())
    predictor_explore.fit(
        train_data=train_data,
        tuning_data=valid_data,
        # hyperparameters=initial_models,
        hyperparameters=auto_ag_priority(final_hyperparameters),
        # hyperparameters=fast,
        # raise_on_no_models_fitted=True
        #   num_gpus=*1
        time_limit=8 * 60 * 60,  # 8小时
    )  # 缩短时间以加速
    leaderboard_explore = predictor_explore.leaderboard(valid_data)
    print(leaderboard_explore)
    best_model_name = leaderboard_explore.iloc[0]["model"]
    print(f"\n[阶段一] 结论: 最佳模型是 {best_model_name}")

    return predictor_explore


# %%
train_ag = TabularDataset(train_data_full)
valid_ag = TabularDataset(valid_data_full)

predictor_explore = run_training_pipeline(train_ag, valid_ag, model_path_suffix=f"{TARGET_LABEL}_full_train")

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

predict_and_save_results(predictor_explore, test_data, test_df, f"{TARGET_LABEL}_test")

