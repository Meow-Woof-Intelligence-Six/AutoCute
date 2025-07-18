#%%
from auto_config import project_dir, qlib_dir, train_dates, valid_dates, test_dates
import qlib
qlib.init(
    provider_uri=qlib_dir,
    region = "cn"
)

# %%
labels = dict(
    Turnover='$Turnover',
    Amplitude='$Amplitude',
    PriceChange='$PriceChange',
    TurnoverRate='$TurnoverRate',
    收盘='$close',
    涨跌幅='$PriceChangePercentage',
    # 涨跌幅='$close/Ref($close, 1) - 1',
)
handler_kwargs = {
    "start_time": train_dates[0],
    "end_time": test_dates[-1],
    "fit_start_time": train_dates[0],
    "fit_end_time": train_dates[-1],
    "instruments": "all",
    "label": (list(labels.values()), 
                list(labels.keys()),

    )
}
from qlib.utils import init_instance_by_config

hd158 = init_instance_by_config(config = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_kwargs
            })
#%%
hd360 = init_instance_by_config(config = {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_kwargs
            })

df158 = hd158.fetch()

df360 = hd360.fetch()
#%%
df158.to_pickle(project_dir / "temp/qlib_alpha158.pkl")
df360.to_pickle(project_dir / "temp/qlib_alpha360.pkl")
#%%

