# %%
from auto_config import project_dir, qlib_dir
import pandas as pd

df158 = pd.read_pickle(project_dir / "temp/qlib_alpha158.pkl")
df360 = pd.read_pickle(project_dir / "temp/qlib_alpha360.pkl")


# %%
# 截面排序
import pandas as pd
import numpy as np
def add_daily_rank_column(
    df: pd.DataFrame, target_column: str, new_column_name: str, 
    ranks_involves_column_name:str, ascending: bool = False
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=[target_column])  # 删除目标列中有NaN的行
    ranks = df_copy.groupby(level=0)[target_column].rank(
        ascending=ascending, method="first"
    )
    df_copy[new_column_name] = ranks
    df_copy[new_column_name] = df_copy[new_column_name].astype(int)

    # 增加一列参与排名的股票数量
    involves_counts = df_copy.groupby(level=0)[target_column].count()
    df_copy[ranks_involves_column_name] = df_copy.index.get_level_values(0).map(involves_counts)


    return df_copy

# 1. 每天不一定有300只股票
# 2. 收盘缺失的数据没有意义。

df158 = add_daily_rank_column(
    df158, target_column="涨跌幅", new_column_name="涨跌幅排名", 
    ranks_involves_column_name="涨跌幅参与排名股票数量", ascending=False
)
df360 = add_daily_rank_column(
    df360, target_column="涨跌幅", new_column_name="涨跌幅排名", 
    ranks_involves_column_name="涨跌幅参与排名股票数量", ascending=False
)
#%%

# 1-300
import pandas as pd
import numpy as np


def reformat_stock_index(df: pd.DataFrame, instrument_col="instrument") -> pd.DataFrame:
    """
    对金融时间序列DataFrame的索引进行重构和格式化。

    该函数执行以下操作：
    1. 调换MultiIndex的顺序，从 (datetime, instrument) 变为 (item_id, timestamp)。
    2. 将 'instrument' 代码（如 'SZ000111'）转换为纯数字整数（如 111）。
    3. 将索引级别名称 'datetime' 重命名为 'timestamp'，'instrument' 重命名为 'item_id'。

    Args:
        df (pd.DataFrame):
            输入的DataFrame。必须满足以下条件：
            - 索引为MultiIndex。
            - 第一个索引级别（level 0）为日期时间对象。
            - 第二个索引级别（level 1）为字符串形式的股票代码。

    Returns:
        pd.DataFrame:
            一个经过重构和格式化的新的DataFrame。
    """
    # 检查输入是否为DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是Pandas DataFrame。")

    # 复制DataFrame以避免修改原始数据
    df_processed = df.copy()

    # 获取原始索引的名称，以便后面引用
    # 如果索引没有名字，reset_index()会默认使用 'level_0', 'level_1'
    original_names = df_processed.index.names
    date_col = original_names[0] if original_names[0] is not None else "level_0"
    instrument_col = original_names[1] if original_names[1] is not None else "level_1"

    # 1. 将索引转为列
    df_processed = df_processed.reset_index()

    # 2. 修改instrument列，提取数字并转为整数
    df_processed[instrument_col] = (
        df_processed[instrument_col].str.extract(r"(\d+)").astype(int)
    )

    # 按照股票代码排序
    df_processed = df_processed.sort_values(by=instrument_col)

    # 3. 重命名列
    df_processed = df_processed.rename(
        columns={date_col: "timestamp", instrument_col: "item_id"}
    )

    # 4. 按新顺序和名称设置索引
    df_final = df_processed.set_index(["item_id", "timestamp"])

    df_final["涨跌正负"] = df_final["涨跌幅"] > 0
    df_final["涨跌"] = df_final["涨跌幅排名"] <= df_final["涨跌幅参与排名股票数量"]/2
    # 比赛求的既是相对10，也是绝对10
    # 绝对10
    # df_final["龙虎"] = (df_final["涨跌幅排名"] < 1 + 10) | (df_final["涨跌幅排名"] > df_final["涨跌幅参与排名股票数量"] - 10) 
    # 相对10
    df_final["龙虎"] = np.minimum(df_final["涨跌幅排名"], df_final["涨跌幅参与排名股票数量"] - df_final["涨跌幅排名"] + 1) <= 10/300* df_final["涨跌幅参与排名股票数量"]



    return df_final

df158 = reformat_stock_index(df158.sort_values(by=["datetime", "涨跌幅排名"]))
df360 = reformat_stock_index(df360.sort_values(by=["datetime", "涨跌幅排名"]))

# %%
# 按照timestamp 排序，然后导出 pkl
df158 = df158.sort_index(level=1)
df360 = df360.sort_index(level=1)
df158.to_pickle(project_dir / "temp/qlib_alpha158_ranked.pkl")
df360.to_pickle(project_dir / "temp/qlib_alpha360_ranked.pkl")
# %%
print("Processing completed successfully!")
print(f"df158 shape: {df158.shape}")
print(f"df360 shape: {df360.shape}")
print(f"Saved files:")
print(f"  - {project_dir / 'temp/qlib_alpha158_ranked.pkl'}")
print(f"  - {project_dir / 'temp/qlib_alpha360_ranked.pkl'}")
# %%
