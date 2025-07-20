尊敬的“2025年中国高校计算机大赛—大数据挑战赛”竞赛主办方：

您好！

我们是参赛团队 “喵汪特工队”。根据竞赛的《代码审核要求》，现将我们在本次比赛中所使用的第三方开源资源报备如下。

所有资源均在2025年5月1日前开源，在huggingface、github等平台上公开发布，我们承诺在后续提交的代码中，所使用的对应文件的MD5值与本次报备完全一致。

一、 数据获取方式说明 (针对动态数据)
本次比赛中，我们使用了开源财经数据接口库 akshare 来获取部分个股补充信息。

所用库的开源链接： https://github.com/akfamily/akshare

关于数据文件及MD5值的说明：

由于此部分数据是通过API动态获取而非下载单个静态文件，我们采取以下方式确保可复现性：

我们已将通过 akshare 在特定时间点获取的全部数据，保存为了一个本地的 Joblib 文件。该文件已包含在我们的代码提交包中，其MD5值如下：

文件名及MD5值：

256f95ace851315c6f38c21f0b4c31be all_stock_info_df.joblib

c1184b6f1bb05d85846a52654263a619 all_stock_info_df_xq.joblib

688b5b59e6617a2265b17d1c28849390 沪深A股_2015Q1_2025Q1_利润表.csv

82739848c03e12e4334d873e35318a54 沪深A股_2015Q1_2025Q1_财报数据.csv

b6cf983c19b3897bdc2e577ad363bee9 沪深A股_2015Q1_2025Q1_资产负债表.csv

二、 使用的开源预训练模型
1. 模型名称： TabPFN (Foundation Model for Tabular Data)

开源链接地址： https://github.com/PriorLabs/TabPFN

文件名及MD5值：

16bdb6b7041d4cbbe81f7d9153b00422 tabpfn-v2-classifier.ckpt

13bd1cee380f5ac2ec1da942bba7a910 tabpfen-v2-regressor.ckpt

2. 模型名称： TabPFNMix Classifier

开源链接地址： https://huggingface.co/autogluon/tabpfn-mix-1.0-classifier

说明： 此模型包含多个文件，我们对整个模型目录进行打包计算MD5值。

获取方式及MD5值：

运行命令 tar -cf - data/pretrained/autogluon/tabpfn-mix-1.0-classifier | md5sum

输出结果: 8d4b680dd7e98e0c859d8f868cef8d4b -

3. 模型名称： AutoGluon - Chronos 系列模型

开源链接地址： https://huggingface.co/collections/amazon/chronos-models-and-datasets-65f1791d630a8d57cb718444

模型文件及MD5值说明： 由于我们使用了 Chronos 系列的多个预训练模型，为每一个模型文件单独提供MD5值不切实际。为保证复现的准确性，我们提供以下可复现的批量下载方式。

依赖库版本： huggingface-hub==0.23.4

复现下载脚本：

from huggingface_hub import snapshot_download

model_ids = [
    "amazon/chronos-t5-tiny",
    "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
]

for model_id in model_ids:
    print(f"Downloading {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"./models/{model_id.replace('/', '_')}",
        local_dir_use_symlinks=False
    )
print("All Chronos models downloaded.")

说明： 在指定 huggingface-hub 版本下运行以上Python代码，即可下载我们使用的全部Chronos模型文件。如图所示，从开源链接网页中可以看到，这一系列的模型在2025年5月1日到邮件发送日期期间，并无更新，因而下载的版本是早已发布开源的（2025年2月17日）。

4. 模型名称： 其他使用的文本预训练模型

说明： 本部分列出了我们使用的其他文本类预训练模型，包括 Qwen3, ELECTRA, BERT, LayoutLMv3, T5。与Chronos系列类似，由于文件数量较多，我们统一提供以下可复现的批量下载方式。

合规性说明：

Qwen/Qwen3-0.6B 模型于2025年4月29日发布并官方开源权重(https://qwenlm.github.io/zh/blog/qwen3/)，符合“5月1日前开源”的要求。我们使用的是社区转换的Hugging Face格式。

其余模型均为早已公开发布的经典模型。

依赖库版本： huggingface-hub==0.23.4

复现下载脚本：

from huggingface_hub import snapshot_download

# List of other text models used
text_model_ids = [
    "Qwen/Qwen3-0.6B",
    "google/electra-base-discriminator",
    "bert-base-cased",
    "microsoft/layoutlmv3-base",
    "t5-small",
]

for model_id in text_model_ids:
    print(f"Downloading {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"./models/{model_id.replace('/', '_')}", # 指定下载目录
        local_dir_use_symlinks=False # 建议设为False，避免符号链接问题
    )
print("All other text models downloaded.")

三、 使用的开源框架内模型 (仅作说明)
为确保信息的完整性，我们在此说明，除了上述需要报备的预训练模型外，我们还利用了多个框架中的多种模型/算法/特征工程处理算法。这些模型并非预训练模型，而是在竞赛提供的 train.csv 数据集上从零开始训练的。因此，根据规则，我们理解这部分无需作为预训练模型进行报备。

感谢主办方的辛勤工作。如对报备内容有任何疑问，请随时与我们联系。



顺祝商祺！

“喵汪特工队” 团队 队长叶璨铭 2025年7月18日