#%%
from auto_config import project_dir, qlib_dir, train_dates, valid_dates, test_dates

#%%
segments = dict(
    train=train_dates,
    valid=valid_dates,
    test=test_dates,
)
import qlib
qlib.init(
    provider_uri=qlib_dir,
    region = "cn"
)
from qlib.data import D
# %%
# universe = D.list_instruments(D.instruments("all"), start_time=train_dates[0], end_time=test_dates[1])

# %%
instrument_col = "instrument"
# 截面排序
import pandas as pd
import numpy as np

def add_daily_rank_column(
    df: pd.DataFrame, 
    target_column: str, 
    new_column_name: str, 
    ascending: bool = False
) -> pd.DataFrame:
    """
    对DataFrame按日进行截面排序，并添加一个新的排序列。

    函数会根据指定的列（target_column），在每个交易日内对所有股票进行排序，
    并将排名结果存储在一个新的列（new_column_name）中。

    Args:
        df (pd.DataFrame): 
            输入的DataFrame，索引必须是 (datetime, instrument) 的MultiIndex。
        target_column (str): 
            用于排序的现有列的名称，例如 'price' 或 'volume'。
        new_column_name (str): 
            要创建的新排序列的名称，例如 'volume_rank'。
        ascending (bool, optional): 
            排序方式。
            - False (默认): 降序排序，值越大排名越靠前（即排名为1）。适用于成交量、涨幅等。
            - True: 升序排序，值越小排名越靠前（即排名为1）。适用于市盈率等。

    Returns:
        pd.DataFrame: 
            一个包含新排序列的DataFrame副本。

    Raises:
        ValueError: 如果指定的 `target_column` 或 `new_column_name` 无效。
        TypeError: 如果输入的不是DataFrame。
    """
    # --- 输入校验 ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是Pandas DataFrame。")
    if target_column not in df.columns:
        raise ValueError(f"错误: 用于排序的列 '{target_column}' 不存在于DataFrame中。")
    if new_column_name in df.columns:
        raise ValueError(f"错误: 新列名 '{new_column_name}' 已存在，请使用其他名称。")

    # 复制DataFrame以避免修改原始数据
    df_copy = df.copy()

    # ---核心逻辑---
    # 1. 按索引的第一个级别（即datetime）进行分组
    # 2. 选择要排序的目标列
    # 3. 在每个组内调用 .rank() 方法进行排序
    #    method='first' 用于在处理相同值时，根据其在原数据中的顺序分配唯一排名，避免排名重复
    ranks = df_copy.groupby(level=0)[target_column].rank(ascending=ascending, method='first')

    # 4. 将计算出的排名序列作为新列添加到DataFrame中
    df_copy[new_column_name] = ranks
    
    # 将新列的数据类型转为整数，使显示更整洁
    df_copy[new_column_name] = df_copy[new_column_name].astype(int)

    return df_copy



# 1-300
import pandas as pd
import numpy as np

def reformat_stock_index(df: pd.DataFrame) -> pd.DataFrame:
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
    date_col = original_names[0] if original_names[0] is not None else 'level_0'
    instrument_col = original_names[1] if original_names[1] is not None else 'level_1'

    # 1. 将索引转为列
    df_processed = df_processed.reset_index()
    


    # 2. 修改instrument列，提取数字并转为整数
    df_processed[instrument_col] = df_processed[instrument_col].str.extract(r'(\d+)').astype(int)

    # 按照股票代码排序
    df_processed = df_processed.sort_values(by=instrument_col)

    # 3. 重命名列
    df_processed = df_processed.rename(columns={
        date_col: 'timestamp',
        instrument_col: 'item_id'
    })

    # 4. 按新顺序和名称设置索引
    df_final = df_processed.set_index(['item_id', 'timestamp'])

    return df_final

#%%
def pipeline(df, return_ag=False):
    df.columns = [col[1] for col in df.columns.values]
    df = add_daily_rank_column(
    df, 
    target_column='涨跌幅', 
    new_column_name='涨跌幅排名'
    )
    df = df.sort_values(by=['datetime', '涨跌幅排名'])
    df = reformat_stock_index(df)

    df['涨跌'] = df['涨跌幅']>0
    df['龙虎'] = (df['涨跌幅排名']<=0+10)|(df['涨跌幅排名']>=301-10)
    if return_ag:
        from autogluon.timeseries import TimeSeriesDataFrame
        tsdf = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp",
            # static_features_df=static_features_df, #TODO
            # known_covariates_names=[] # TODO 如假期
        )
        return tsdf
    else:
        return df
#%%
# %%
from qlib.data.dataset import DatasetH, TSDatasetH
labels = dict(
    Turnover='$Turnover',
    Amplitude='$Amplitude',
    PriceChange='$PriceChange',
    TurnoverRate='$TurnoverRate',
    收盘='$close',
    涨跌幅='$PriceChangePercentage',
    # 涨跌幅='$close/Ref($close, 1) - 1',
    # 涨跌='$close>Ref($close, 1)',
    # 涨跌排名='Rank($close/Ref($close, 1) - 1, 0)',
    # 前十
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

hd_158 = init_instance_by_config(config = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_kwargs
            })
hd_360 = init_instance_by_config(config = {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_kwargs
            })

# df = hd_158.fetch()
# #%%
# hd_360 = init_instance_by_config(config = {
#                 "class": "Alpha360",
#                 "module_path": "qlib.contrib.data.handler",
#                 "kwargs": handler_kwargs
#             })
# df360 = hd_360.fetch()
# #%%
# df.to_pickle(project_dir / "temp/qlib_alpha158.pkl")
# df360.to_pickle(project_dir / "temp/qlib_alpha360.pkl")
# %%
def tsdf_from_handler(handler):
    dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler,
            "segments": {
                "train": train_dates,
                "valid": valid_dates,
                "test": test_dates,
            },
        },
    }
    dataset = init_instance_by_config(dataset_conf)
    df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=handler.DK_L)
    df_valid =dataset.prepare("valid", col_set=["feature", "label"], data_key=handler.DK_L)
    df_test =dataset.prepare("test", col_set=["feature", "label"], data_key=handler.DK_I)
    tsdf_train = pipeline(df_train)
    tsdf_valid = pipeline(df_valid)
    tsdf_test = pipeline(df_test)
    return tsdf_train, tsdf_valid, tsdf_test
# %%
tsdf_train, tsdf_valid, tsdf_test =tsdf_from_handler(hd_158)
#%%
tsdf_train360, tsdf_valid360, tsdf_test360 =tsdf_from_handler(hd_360)
#%%
