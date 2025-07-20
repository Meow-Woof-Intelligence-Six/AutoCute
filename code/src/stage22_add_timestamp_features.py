# %%
import pandas as pd
from auto_config import project_dir

# df158 = pd.read_pickle(project_dir / "temp/lag158.pkl")
df158 = pd.read_pickle(project_dir / "temp/lag158_finance.pkl")

# %%
# https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html

# %%
from feature_engine.datetime import DatetimeFeatures

date_features = DatetimeFeatures(
    variables="timestamp",
    features_to_extract=[
        "year",
        "quarter",
        "month",
        "week",
        "day_of_month",
        "day_of_week",
        "day_of_year",
        "month_end",
        "quarter_end",
    ],
    drop_original=False
    # drop_original=True,  # 删除原始的时间戳列 
    # 不能删，后面还需要train_test_split
)
df158 = date_features.fit_transform(df158)
df158
#%%

from feature_engine.creation import CyclicalFeatures
# 需编码的周期性特征（月、周、日）
cyclical_vars = ['timestamp_month', 'timestamp_day_of_week', 'timestamp_day_of_month']
# 手动指定周期（金融场景中关键！）
cyclical_encoder = CyclicalFeatures(
    variables=cyclical_vars,
    max_values={
        'timestamp_month': 12,          # 12个月
        'timestamp_day_of_week': 6,    # 0-6（周一=0）
        'timestamp_day_of_month': 31   # 最大31天
    },
    drop_original=False  # 保留原始特征（后续可筛选）
)

df158 = cyclical_encoder.fit_transform(df158)
df158
# %%
# df158.to_pickle(project_dir / "temp/lag158_with_timestamp_features.pkl")
df158.to_pickle(project_dir / "temp/lag158_finance_with_timestamp_features.pkl")

# %%
# 保存 特征转换器
import joblib
time_features_maker = dict(
    date_features=date_features,
    cyclical_encoder=cyclical_encoder
)
time_features_maker_path = project_dir / "temp/stage2/time_features_maker.pkl"
time_features_maker_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(time_features_maker, time_features_maker_path)


# %%
