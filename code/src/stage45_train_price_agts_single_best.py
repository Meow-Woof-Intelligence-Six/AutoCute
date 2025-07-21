# %%
from stage35_simple_price_data import TARGET_LABEL, data, project_dir
from auto_config import pretrained_dir
# metric_name = "MASE"
metric_name = "RMSSE"
prediction_length = 3

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    # target="收盘",
    target=TARGET_LABEL,
    eval_metric=metric_name,  # Primary metric for training
    verbosity=4,
    freq="D",
    path=project_dir
    / "model/stage45_train_price_agts_single_explore",  # Save model to this path
)
explore_hyperparameters = {
    "TemporalFusionTransformer": [
        {
            "ag_args": {"name_suffix": "Default"},
        },
        # {
        #     "ag_args": {"name_suffix": "FineTuned"},
            
        # },
    ],
    "RecursiveTabular":{}, 
    "DirectTabular":{}, 
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
    "TiDE": {
        "encoder_hidden_dim": 256,
        "decoder_hidden_dim": 256,
        "temporal_hidden_dim": 64,
        "num_batches_per_epoch": 100,
        "lr": 1e-3,
    },
    "DeepAR": {},
    "PatchTST": {}, 
    # "AutoARIMA": {}, # 0.34
    # "AutoETS": {},  # best quality 0.56
    # "DynamicOptimizedTheta": {},  # best quality 0.02
    # "SeasonalNaive": {},  # best quality 0.07



}

predictor.fit(
    refit_full=True,
    num_val_windows=2,
    train_data=data,
    random_seed=2002,
    hyperparameters=explore_hyperparameters
)
