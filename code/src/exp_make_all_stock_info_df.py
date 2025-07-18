#%%
from auto_config import qlib_dir

all_text = qlib_dir / "instruments/all.txt"
with open(all_text, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    instruments = [line.strip().split()[0] for line in lines]
instruments
# %%
import akshare as ak
import pandas as pd
import random
from tqdm import tqdm

# 假设这是包含300个股票代码的列表

stock_info_list = []
for symbol in tqdm(stock_list):
    try:
        stock_individual_info_em_df = ak.stock_individual_info_em(symbol=symbol)
        transposed_df = stock_individual_info_em_df.set_index('item').T.reset_index(drop=True)
        stock_info_list.append(transposed_df)
    except Exception as e:
        print(f"获取股票代码 {symbol} 信息失败: {e}")

if stock_info_list:
    all_stock_info_df = pd.concat(stock_info_list, ignore_index=True)
else:
    all_stock_info_df = pd.DataFrame()
    print("未能获取任何股票信息。")


# TODO
# import akshare as ak

# stock_zyjs_ths_df = ak.stock_zyjs_ths(symbol="000066")
# print(stock_zyjs_ths_df)



# TODO 财报
# import akshare as ak

# stock_zygc_em_df = ak.stock_zygc_em(symbol="SH688041")
# print(stock_zygc_em_df)


all_stock_info_df
#%%
# TODO 主营构成
import akshare as ak

# stock_zygc_em_df = ak.stock_zygc_em(symbol="SH688041")
stock_zygc_em_df = ak.stock_zygc_em(symbol="SZ000001")
stock_zygc_em_df
#%%
# TODO A股质押，7天一次。
import akshare as ak

stock_gpzy_profile_em_df = ak.stock_gpzy_profile_em()
stock_gpzy_profile_em_df

# %%
# 文本新闻，每日
import akshare as ak

stock_gsrl_gsdt_em_df = ak.stock_gsrl_gsdt_em(date="20250718")
print(stock_gsrl_gsdt_em_df)

# 接待机构调研
import akshare as ak

stock_jgdy_tj_em_df = ak.stock_jgdy_tj_em(date="20210128")
print(stock_jgdy_tj_em_df)

import akshare as ak

# news_report_time_baidu_df = ak.news_report_time_baidu(date="20250718")
news_report_time_baidu_df = ak.news_report_time_baidu(date="20250717")
news_report_time_baidu_df = ak.news_report_time_baidu(date="20250716")
news_report_time_baidu_df

#%%

#%%
if not all_stock_info_df.empty:
    prompt_templates = [
        # 原始中文模板 1
        "预测任务为{股票简称}（股票代码：{股票代码}）的每日涨跌幅排名的相对值。其中，涨跌幅定义为当日收盘价相对于前一日收盘价的变动比例：$close/Ref(close, 1)-1。涨跌幅排名是指在包含300只股票的股票池中，根据涨跌幅的大小进行排序，排名范围为1-300，数值越小表示涨幅越大。涨跌幅排名相对值是指对原始涨跌幅排名进行quantile变换，使其转换成服从均值为0、方差为1的正态分布数值。股票所属行业为{行业}。",
        # 英文翻译 1
        "The prediction task is to forecast the daily relative ranking of price change for {股票简称} (stock code: {股票代码}). Price change is defined as $close/Ref(close, 1)-1. The ranking among 300 stocks is sorted by this change from 1 (highest increase) to 300 (largest decrease). The relative ranking value is obtained by applying a quantile transform to the original ranking to follow a normal distribution with mean 0 and variance 1. The industry of the stock is {行业}.",
        # 原始中文模板 2
        "请预测股票代码为{股票代码}（股票简称：{股票简称}）的每日涨跌幅排名相对值。每日涨跌幅计算公式为：$close/Ref(close, 1)-1。该排名是基于300只股票的涨跌幅表现，数值1代表涨幅最大，300代表跌幅最大。最终目标是预测经过quantile变换后得到的相对排名值，该值服从均值为0、方差为1的正态分布，数值越小意味着原始涨幅排名更高。该股票上市时间为{上市时间}。",
        # 英文翻译 2
        "Please predict the daily relative ranking value of price change for stock code {股票代码} (name: {股票简称}). Daily price change is calculated as $close/Ref(close, 1)-1. The ranking among 300 stocks ranges from 1 (largest increase) to 300 (largest decrease). The target is the relative ranking value obtained after a quantile transform, following a normal distribution with mean 0 and variance 1, where smaller values indicate higher original ranking. The stock's listing date is {上市时间}.",
        # 原始中文模板 3
        "时序预测任务：预测{股票代码} - {股票简称}的每日涨跌幅排名相对值。涨跌幅定义：(今日收盘价 / 昨日收盘价) - 1。排名规则：300只股票中，涨幅最大者排名1，跌幅最大者排名300。相对值定义：对排名进行quantile标准化，使其均值为0，方差为1，数值小者表示涨幅靠前。最新信息显示，该股票最新价格为{最新}。",
        # 英文翻译 3
        "Time-series forecasting task: predict the daily relative ranking value of price change for {股票代码} - {股票简称}. Price change is defined as (today's close / yesterday's close) - 1. In 300 stocks, the largest increase is rank 1 and the largest decrease is rank 300. The relative value is obtained by quantile standardization of the ranking to mean 0 and variance 1, where smaller values indicate higher increases. The latest price is {最新}.",
        # 原始中文模板 4
        "针对{行业}行业的股票{股票简称}（{股票代码}），进行每日涨跌幅排名的相对值预测。涨跌幅计算：$close(t) / close(t-1) - 1$。在300只股票中进行排名（1-300，1为最优涨幅），然后通过quantile变换得到均值为0、方差为1的相对排名值（数值越小，涨幅越大）。",
        # 英文翻译 4
        "For the {行业} sector, forecast the daily relative ranking value of price change for {股票简称} ({股票代码}). Price change is calculated as $close(t) / close(t-1) - 1$. Among 300 stocks, ranks range from 1 (best increase) to 300 (worst). Then apply a quantile transform to get values with mean 0 and variance 1, where smaller values indicate larger increases.",
        # 原始中文模板 5
        "预测任务：{股票代码} ({股票简称}) 的每日股价变动相对排名。变动指标为每日收盘价相对于前一日收盘价的百分比变化。在300只股票中，根据此百分比变化进行排名（1代表最高涨幅，300代表最大跌幅）。最后，将此排名通过quantile转换，得到一个均值为0、方差为1的数值，数值越低表示涨幅越大。",
        # 英文翻译 5
        "Prediction task: daily relative ranking of price change for {股票代码} ({股票简称}). The metric is daily close price percentage change relative to the previous close. Rank among 300 stocks (1 is highest increase, 300 is largest decrease). Finally, apply a quantile transform to the ranking to obtain values with mean 0 and variance 1, where lower values indicate larger increases."
    ]

    prompts = []
    for index, row in all_stock_info_df.iterrows():
        prompt_template = random.choice(prompt_templates)
        prompt = prompt_template.format(
            最新=row.get('最新', 'N/A'),
            股票代码=row.get('股票代码', 'N/A'),
            股票简称=row.get('股票简称', 'N/A'),
            总股本=row.get('总股本', 'N/A'),
            流通股=row.get('流通股', 'N/A'),
            总市值=row.get('总市值', 'N/A'),
            流通市值=row.get('流通市值', 'N/A'),
            行业=row.get('行业', 'N/A'),
            上市时间=row.get('上市时间', 'N/A')
        )
        prompts.append(prompt)

    all_stock_info_df['prompt'] = prompts
    print(all_stock_info_df[['股票代码', '股票简称', 'prompt']])
else:
    print("没有可生成prompt的股票信息。")
#%%
import joblib
joblib.dump(all_stock_info_df, project_dir / "data/additional/all_stock_info_df.joblib")
# %%


#%%
import akshare as ak

stock_gpzy_pledge_ratio_detail_em_df = ak.stock_gpzy_pledge_ratio_detail_em()
(stock_gpzy_pledge_ratio_detail_em_df)
# %%
# 只有一天，不能用
import akshare as ak

stock_gpzy_industry_data_em_df = ak.stock_gpzy_industry_data_em()
(stock_gpzy_industry_data_em_df)
# %%

# 一年一次的数据，和股票无关
# 反映A股整体情况
import akshare as ak

stock_sy_profile_em_df = ak.stock_sy_profile_em()
(stock_sy_profile_em_df)
# %%
# 处理较为复杂。 很久一次
import akshare as ak

# stock_sy_yq_em_df = ak.stock_sy_yq_em(date="20221231")
stock_sy_yq_em_df = ak.stock_sy_yq_em(date="20250331")
(stock_sy_yq_em_df)
# import akshare as ak

# stock_sy_em_df = ak.stock_sy_em(date="20250718")
# print(stock_sy_em_df)
# %%
# 每年、每个行业，多个分析师评级
import akshare as ak

stock_analyst_rank_em_df = ak.stock_analyst_rank_em(year='2024')
print(stock_analyst_rank_em_df)

#%%
# 每一个分析师，隔一段时间，说买入一个。
# import akshare as ak

# stock_analyst_detail_em_df = ak.stock_analyst_detail_em(analyst_id="11000200926", indicator="最新跟踪成分股")
# print(stock_analyst_detail_em_df)
# %%
# 我们的目标
# 财报，几个月更新一次，7.20-


#%%
# 每天 股票，机构参与度。
stock_comment_detail_zlkp_jgcyd_em_df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol="600000")
print(stock_comment_detail_zlkp_jgcyd_em_df)

# 每天，股票，平分、参与意愿
# stock_comment_detail_zhpj_lspf_em_df = ak.stock_comment_detail_zhpj_lspf_em(symbol="600000")
# print(stock_comment_detail_zhpj_lspf_em_df)

#%%
# 财报数据
# 筛选
# https://akshare.akfamily.xyz/data/stock/stock.html#id133