# %%
from stage35_simple_price_data import TARGET_LABEL, data, project_dir

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
model_path = project_dir / "model/stage45_train_price_agts_stats"
predictor = TimeSeriesPredictor.load(model_path)
# %%
predictions = predictor.predict(data)
predictions
#%%
from auto_config import date_before_test, test_date
from stage59 import predictions_to_competition_df
model_mode = "price"
model_name = predictor.path.split("/")[-1]
result_df = predictions_to_competition_df(
    predictions,
    test_date = test_date,
    date_before_test=date_before_test,
    test_data_for_autogluon=data,
    model_mode=model_mode
)
result_df
# %%
result_df.to_csv(
    project_dir / f"output/result-{model_mode}-{model_name}.csv", index=False
)

# %%
