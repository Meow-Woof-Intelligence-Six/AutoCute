# %%
from stage35_simple_price_data import TARGET_LABEL, data, project_dir

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
# model_path = project_dir / "model/stage45_train_price_agts_stats"
model_path = project_dir / "model/stage45_train_price_agts_single_explore"
predictor = TimeSeriesPredictor.load(model_path)
# %%
predictions = predictor.predict(data)
predictions
#%%
# Remove the last date from data
data1 = data[data.index.get_level_values('timestamp') < data.index.get_level_values('timestamp').max()]
predictions1 = predictor.predict(data1)
predictions1
#%%
from auto_config import test1_dates, test_dates
from stage59 import predictions_to_competition_df
from datetime import datetime, timedelta
real_test_date = datetime.strftime(datetime.strptime(test_dates[0], '%Y-%m-%d') + timedelta(days=3), '%Y-%m-%d')
real_test_date
#%%

model_mode = "price"
model_name = predictor.path.split("/")[-1]
result_df = predictions_to_competition_df(
    predictions,
    test_date = real_test_date,
    date_before_test=test_dates[0],
    test_data_for_autogluon=data,
    model_mode=model_mode
)
result_df
# %%
result_df.to_csv(
    project_dir / f"temp/output/result-{model_mode}-{model_name}-test.csv", index=False
)

# %%
result_df1 = predictions_to_competition_df(
    predictions1,
    test_date = test_dates[0],
    date_before_test=test1_dates[0],
    test_data_for_autogluon=data,
    model_mode=model_mode
)
result_df1
# %%
result_df1.to_csv(
    project_dir / f"temp/output/result-{model_mode}-{model_name}-test1.csv", index=False
)
# %%
