#%%
from auto_config import project_dir, csv_path, qlib_dir, train_dates, valid_dates, test_dates
import pandas as pd
# 百万数据量 365*10*300 = 1095000
# https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format
column_mapping = {
        "股票代码": "symbol",
        "日期": "date",
        # OCHLV
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        # Other features
        "成交额": "Turnover",
        "振幅": "Amplitude",
        "涨跌额": "PriceChange",
        "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage", # 百分比
    }

df = pd.read_csv(csv_path, header=0, sep=",", encoding="utf-8")
df.rename(columns=column_mapping, inplace=True)

df['factor'] = 1.0  # 添加 赋权，当做没有，初始值为1.0
len(df)  # 查看数据量


df['vwap'] = df['Turnover'] / df['volume']  # 计算 VWAP

# 637229；有些股票中间才出现。
#%%
# TODO 复权价格问题
# df['close'].describe()
# bad = df[df['close'] <= 0]
# bad[['symbol', 'date', 'close']]
# df = df[(df['收盘'] <= 0) | (df['开盘'] <= 0) | (df['最高'] <= 0) & (df['最低'] <= 0)]

#%%
# 转换日期格式
df['date'] = pd.to_datetime(df['date'])

# 股票代码格式化为6位数字，前面补0
df['symbol'] = df['symbol'].astype(str).str.zfill(6)
#%%
# 不应该开启。
# 没开启的情况下，缺失是因为统计指标没到天数、股票还没上市等原因。
# 开启了认为增加了很多行。
补充日期 = False
temp_qlib_import_csvs_path = project_dir/"temp/qlib_import_csvs"
temp_qlib_import_csvs_path.mkdir(parents=True, exist_ok=True)

# 按symbol分组并保存为单独的CSV文件
for symbol, group_df in df.groupby('symbol'):
    csv_file_path = temp_qlib_import_csvs_path / f"{symbol}.csv"
    if 补充日期:
        full_range = pd.date_range(group_df['date'].min(), group_df['date'].max(), freq='D')
        full_df = pd.DataFrame({'date': full_range})
        full_df['symbol'] = symbol
        merged = pd.merge(full_df, group_df, on=['symbol', 'date'], how='left', sort=True)
    else: 
        merged = group_df
    merged.to_csv(csv_file_path, index=False, header=True, sep=",", mode="w", encoding="utf-8")


#%%
from qlib_scripts.dump_bin import DumpDataAll
dumper = DumpDataAll(
    csv_path=temp_qlib_import_csvs_path.as_posix(),
    qlib_dir=qlib_dir.as_posix(),
    symbol_field_name="symbol",
    date_field_name ="date",
)
dumper.dump()

# %%
