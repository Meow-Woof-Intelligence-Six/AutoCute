#%%

from pathlib import Path
from tempfile import tempdir
this_file = Path(__file__).resolve()
this_dir = this_file.parent
code_dir = this_dir.parent
project_dir = code_dir.parent
project_dir

#%%
# 官方的数据目录
data_dir = project_dir / "data"
model_dir = project_dir / "model"
temp_dir = project_dir / "temp"

# 报备的数据目录
custom_data_dir = project_dir / "code/data"

pretrained_dir = custom_data_dir / "pretrained"

#%%

csv_path = project_dir/"data/test.csv"
qlib_dir = project_dir/"temp/qlib_data"

train_csv_path = project_dir/"data/train.csv"
valid_csv_path = project_dir/"data/test.csv"

#%%

# 自动决定 train valid, test 分区
import pandas as pd
from joblib import Memory

# Create a memory cache in the project directory
memory = Memory(location=project_dir / "joblib_cache", verbose=0)

today = pd.to_datetime("today").strftime('%Y-%m-%d')

# --- 可配置参数 ---
# 定义验证集的天数
VALID_DAYS = 5

@memory.cache
def get_train_valid_test_dates(today=today,):
    """
    根据官方数据文件，智能推导训练、验证和测试集的日期范围。
    核心逻辑：将官方数据的最后一天作为测试集，并依此向前推导。
    """
    print("正在重新计算训练/验证/测试集的日期范围...")
    
    # 1. 合并数据源以获取完整日期序列
    train_df_raw = pd.read_csv(train_csv_path, usecols=['日期'], encoding="utf-8")
    valid_df_raw = pd.read_csv(valid_csv_path, usecols=['日期'], encoding="utf-8")
    all_dates_df = pd.concat([train_df_raw, valid_df_raw]).drop_duplicates()
    all_dates_df['日期'] = pd.to_datetime(all_dates_df['日期'])
    
    # 2. 确定测试集
    test_date = all_dates_df['日期'].max()
    test = (test_date.strftime('%Y-%m-%d'), test_date.strftime('%Y-%m-%d'))

    # 2.1 确定第一步测试集
    test_date1 = test_date - pd.Timedelta(days=1)
    test1 = (test_date1.strftime('%Y-%m-%d'), test_date1.strftime('%Y-%m-%d'))

    # 3. 确定验证集
    valid_end_date = test_date - pd.Timedelta(days=1)
    valid_start_date = valid_end_date - pd.Timedelta(days=VALID_DAYS - 1)
    valid = (valid_start_date.strftime('%Y-%m-%d'), valid_end_date.strftime('%Y-%m-%d'))
    
    # 4. 确定训练集
    train_end_date = valid_start_date - pd.Timedelta(days=1)
    train_start_date = all_dates_df['日期'].min()
    train = (train_start_date.strftime('%Y-%m-%d'), train_end_date.strftime('%Y-%m-%d'))
    
    print(f"日期范围计算完成: Train={train}, Valid={valid}, Test1={test1}, Test={test}")
    return train, valid, test1, test

# 调用函数
train_dates, valid_dates, test1_dates, test_dates = get_train_valid_test_dates(today=today)
