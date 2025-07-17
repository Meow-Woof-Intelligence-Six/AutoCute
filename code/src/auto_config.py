#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_dir = this_file.parent
code_dir = this_dir.parent
project_dir = code_dir.parent
project_dir

#%%
csv_path = project_dir/"data/test.csv"
qlib_dir = project_dir/"temp/qlib_data"

train_csv_path = project_dir/"data/train.csv"
valid_csv_path = project_dir/"data/test.csv"
test_length = 3
#%%
# 自动决定 train valid, test 分区
import pandas as pd
from joblib import Memory

# Create a memory cache in the project directory
memory = Memory(location=project_dir / "joblib_cache", verbose=0)

today = pd.to_datetime("today").strftime('%Y-%m-%d')

@memory.cache
def get_train_valid_test_dates(today):
    train_df = pd.read_csv(train_csv_path, header=0, sep=",", encoding="utf-8")
    valid_df = pd.read_csv(valid_csv_path, header=0, sep=",", encoding="utf-8")
    train = (train_df['日期'].min(), train_df['日期'].max())
    
    valid_start = max(pd.to_datetime(train[1]) + pd.Timedelta(days=1), pd.to_datetime(valid_df['日期'].min())).strftime('%Y-%m-%d')
    valid = (valid_start, valid_df['日期'].max())

    test_start = pd.to_datetime(valid[1]) + pd.Timedelta(days=1)
    test_end = test_start + pd.Timedelta(days=test_length - 1)

    test = (test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

    return train, valid, test

train_dates, valid_dates, test_dates = get_train_valid_test_dates(today=today)
train_dates, valid_dates, test_dates
# 左右包含
#%%


#%%
