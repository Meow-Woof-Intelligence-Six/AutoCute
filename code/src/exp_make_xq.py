#%%
from auto_config import qlib_dir

all_text = qlib_dir / "instruments/all.txt"
with open(all_text, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    instruments = [line.strip().split()[0] for line in lines]
instruments
#%%
def add_stock_prefix(code):
    """
    为股票代码自动添加交易所前缀
    参数:
        code (str/int): 股票代码，可以是字符串或整数
    返回:
        str: 带交易所前缀的股票代码 (SHxxxxxx 或 SZxxxxxx)
    """
    # 转换数字为字符串并移除空格
    code_str = str(code).strip()
    
    # 移除已存在的前缀（如果有）
    if code_str.upper().startswith(('SH', 'SZ')):
        code_str = code_str[2:]
    
    # 验证是否为有效股票代码格式
    if not code_str.isdigit() or len(code_str) != 6:
        raise ValueError("无效的股票代码格式: 必须为6位数字")
    
    # 判断交易所类型并添加前缀
    first_char = code_str[0]
    first_two_chars = code_str[:2]
    
    # 上证判定规则
    if first_char in ('6', '9') or first_two_chars == '68':
        return "SH" + code_str
    
    # 深证判定规则
    if first_char in ('0', '3') or first_two_chars in ('00', '30'):
        return "SZ" + code_str
    
    # 无法识别类型
    raise ValueError("无法识别的股票交易所类型")

# %%
import akshare as ak
import pandas as pd
import random
from tqdm import tqdm

# 假设这是包含300个股票代码的列表

stock_info_list = []
for symbol in tqdm(instruments):
    try:
        stock_individual_info_em_df = ak.stock_individual_basic_info_xq(symbol=add_stock_prefix(symbol))
        transposed_df = stock_individual_info_em_df.set_index('item').T.reset_index(drop=True)
        transposed_df['股票代码'] = symbol  # 添加股票代码列
        stock_info_list.append(transposed_df)
    except Exception as e:
        print(f"获取股票代码 {symbol} 信息失败: {e}")

if stock_info_list:
    all_stock_info_df = pd.concat(stock_info_list, ignore_index=True)
else:
    all_stock_info_df = pd.DataFrame()
    print("未能获取任何股票信息。")

#%%
from auto_config import project_dir
import joblib
joblib.dump(all_stock_info_df, project_dir / "data/additional/all_stock_info_df_xq.joblib")
# %%
