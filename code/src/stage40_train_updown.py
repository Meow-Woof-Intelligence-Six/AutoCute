# %%
# 定义目标标签
TARGET_LABEL = "涨跌_shift"
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
train_data_full
# %%
# --- 4. 定义训练函数 ---

from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict

autogluon_models = {
    "FASTAI": [
        {}, 
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
            "emb_drop": 0.4599138419358,
            "epochs": 47,
            "layers": [200, 100],
            "lr": 0.03888383281136287,
            "ps": 0.28193673177122863,
            "ag_args": {"name_suffix": "_r128", "priority": 7},
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
            "bs": 128,
            "emb_drop": 0.12106594798980945,
            "epochs": 38,
            "layers": [200, 100, 50],
            "lr": 0.037991970245029975,
            "ps": 0.33120008492595093,
            "ag_args": {"name_suffix": "_r173", "priority": 8},
        },

    ],
    "XT": [

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
}

# --- 修正后的模型字典 ---
initial_models = {
    LinearModel: {},
    TabPFNV2Model: {},
    **autogluon_models,
}
fast  = {
    "GBM":{}
}
from stage49_infras import auto_ag_priority
predictor_explore = TabularPredictor(
    label=TARGET_LABEL,
    # problem_type=REGRESSION,
    eval_metric="roc_auc",
    path=str(MODEL_OUTPUT_BASE_PATH / "updown_classfication_explore"),
)
predictor_explore.fit(
    train_data=train_data_full,
    tuning_data=valid_data_full,
    hyperparameters=auto_ag_priority(initial_models),
    # hyperparameters=auto_ag_priority(fast),
    num_gpus=1,
    time_limit=60*60*2
)  
# %% --- 5. 定义预测和保存结果的函数 ---
def predict_and_save_results(predictor, test_data, test_df, output_prefix, save_importance=False):
    # 1. 预测概率
    pred_proba = predictor.predict_proba(test_data)

    # 2. 取出 test_df 的前两列
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
predict_and_save_results(predictor_explore, test1_data, test1_df, "updown_test1")

# %% --- 7. 第二次运行：使用 test_data ---
predict_and_save_results(predictor_explore, test_data, test_df, "updown_test")
