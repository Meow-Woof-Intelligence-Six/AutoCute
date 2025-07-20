# %%
from stage35_simple_price_data import TARGET_LABEL, data, project_dir
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
    / "model/stage45_train_price_agts_stats",  # Save model to this path
)

predictor.fit(
    refit_full=True,
    num_val_windows=2,
    train_data=data,
    random_seed=2002,
    hyperparameters={
        # simple
        "Naive": {},
        "SeasonalNaive": {},
        "Average": {},
        "SeasonalAverage": {},
        # statistical
        "AutoETS": {},
        "AutoARIMA": {},
        "AutoCES": {},
        "Theta": {},
        "DynamicOptimizedTheta": {},
        # sparse data
        "NPTS": {},
        "ADIDA": {},
        "CrostonSBA": {},
        "IMAPA": {},
    },
)
