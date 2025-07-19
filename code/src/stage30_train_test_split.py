#%%
# stage30_train_test_split.py
# 职责：加载完整的特征数据，并根据auto_config中定义的日期范围，
# 将其划分为训练、验证、测试三个集合，并分别保存。

import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

#%%
# --- 1. 加载配置 ---
print("--- [1/3] 加载配置 ---")
try:
    # 假设 auto_config.py 能被正确导入
    from auto_config import project_dir, train_dates, valid_dates, test_dates
    print("成功从 auto_config 加载配置。")
    print(f"训练集日期: {train_dates}")
    print(f"验证集日期: {valid_dates}")
    print(f"测试集日期: {test_dates}")
except ImportError:
    # 如果无法导入，则使用备用硬编码值，并给出提示
    print("无法从 auto_config 加载配置，将使用默认路径和日期。")
    project_dir = Path(".")
    train_dates = ('2015-04-20', '2025-04-17')
    valid_dates = ('2025-04-18', '2025-04-24')
    test_dates = ('2025-04-25', '2025-04-25')

# 定义输入和输出路径
INPUT_DATA_PATH = project_dir / "temp/lag158_with_timestamp_features.pkl"
OUTPUT_DIR = project_dir / "temp/stage3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TRAIN_PATH = OUTPUT_DIR / "train.pkl"
OUTPUT_VALID_PATH = OUTPUT_DIR / "valid.pkl"
OUTPUT_TEST_PATH = OUTPUT_DIR / "test.pkl"


#%%
# --- 2. 加载数据 ---
print("\n--- [2/3] 加载完整的特征数据 ---")
print(f"正在从 {INPUT_DATA_PATH} 加载...")
df = pd.read_pickle(INPUT_DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"数据加载完成，共 {len(df)} 行。")
print(f"数据时间范围: {df['timestamp'].min().strftime('%Y-%m-%d')} -> {df['timestamp'].max().strftime('%Y-%m-%d')}")


#%%
# --- 3. 执行划分并保存 ---
print("\n--- [3/3] 执行划分并保存文件 ---")

# 划分训练集
train_df = df[
    (df['timestamp'] >= pd.to_datetime(train_dates[0])) & 
    (df['timestamp'] <= pd.to_datetime(train_dates[1]))
]
train_df.to_pickle(OUTPUT_TRAIN_PATH)
print(f"训练集已保存到 {OUTPUT_TRAIN_PATH}，共 {len(train_df)} 行。")

# 划分验证集
valid_df = df[
    (df['timestamp'] >= pd.to_datetime(valid_dates[0])) & 
    (df['timestamp'] <= pd.to_datetime(valid_dates[1]))
]
valid_df.to_pickle(OUTPUT_VALID_PATH)
print(f"验证集已保存到 {OUTPUT_VALID_PATH}，共 {len(valid_df)} 行。")

# 划分测试集
# 注意：测试集可能只有一天
test_df = df[
    (df['timestamp'] >= pd.to_datetime(test_dates[0])) & 
    (df['timestamp'] <= pd.to_datetime(test_dates[1]))
]
test_df.to_pickle(OUTPUT_TEST_PATH)
print(f"测试集已保存到 {OUTPUT_TEST_PATH}，共 {len(test_df)} 行。")

print("\n--- 数据划分流程全部完成！ ---")

# %%
