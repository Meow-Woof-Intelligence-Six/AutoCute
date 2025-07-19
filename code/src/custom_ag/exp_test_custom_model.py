# %%
%load_ext autoreload
%autoreload 2
#%%
from autogluon.tabular import TabularDataset

train_data = TabularDataset('/home/ye_canming/repos/novelties/ts/comp/stork_predict_hyh/alpha158_train_converted.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('/home/ye_canming/repos/novelties/ts/comp/stork_predict_hyh/alpha158_valid_converted.csv')  # another Pandas DataFrame
#%%
# 按照timestamp列进行筛选
import pandas as pd

# 假设 df 是原始 DataFrame
# 列：['timestamp', 'symbol', ...]
# N 是你希望保留的最近样本数

# N = 1000   # 举例
N = 100   # 举例

# 确保时间戳列是 datetime 类型

# 步骤 1~4
train_data_recent = (
    train_data
    .sort_values('timestamp')          # 先整体升序
    .groupby('item_id', group_keys=False)
    .apply(lambda g: g.tail(N))        # 每组取最近 N 行
    .reset_index(drop=True)
)

test_data_recent = (
    test_data
    .sort_values('timestamp')          # 先整体升序
    .groupby('item_id', group_keys=False)
    .apply(lambda g: g.tail(5))        # 每组取最近 N 行
    .reset_index(drop=True)
)

#%%
label = '龙虎'  # specifies which column do we want to predict
# label = '涨跌'  # specifies which column do we want to predict
# train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

# train_data.head(5)

#%%
import sys
from pathlib import Path
this_file = Path(__file__).resolve()
this_dir = this_file.parent
sys.path.append(str(this_dir))
project_dir = this_dir.parent.parent
#%%


from autogluon.tabular import TabularPredictor
from ag_svm import AgSVMModel
from ag_tabpfn import TabPFNModel
from ag_linear import IntelligentLinearModel
from ag_nb import IntelligentNaiveBayesModel
from autogluon.tabular.models.lr.lr_model import LinearModel
# %%
# from autogluon.tabular.models.tabpfn.tabpfn_model import TabPFNModel 
# 配置包含SVM的超参数
hyperparameters = {
    # AgSVMModel: [
    #     {}, 
    # ],
    #     {
    #         'gamma': 'auto',
    #     }, 
    #     {
    #         'kernel': 'linear',
    #         'gamma': 'auto',
    #     }, 
    #     {
    #         'kernel': 'poly',
    #         'gamma': 'auto',
    #     },
    #     {
    #         'kernel': 'sigmoid',
    #         'gamma': 'auto',
    #     }
    # ],
    TabPFNModel: {},
    # IntelligentLinearModel:{},
    # LinearModel:{} # 0.9619
    # IntelligentNaiveBayesModel: {},
    # AssertionError: Max allowed features for the model is 100
}

# 初始化预测器
predictor = TabularPredictor(label=label, 
                            #  eval_metric="roc_auc", 
                             problem_type="regression"
                             ).fit(
    # train_data=train_data,
    # tuning_data=test_data,
    train_data=train_data_recent,
    tuning_data=test_data_recent,
    hyperparameters=hyperparameters,
    time_limit=120,   # 2分钟训练
    num_gpus=1
)
# thunder
# 65.39s
# 0.5473
# 16884.2 rows/s
# sklearn
# 106.2s
# 0.5473
# 4603.9 rows/s
# 查看模型表现
#%%
leaderboard = predictor.leaderboard()
leaderboard
# %%
# svr 官方实例
import numpy as np
from sklearn.datasets import make_regression
from thundersvm import SVR

print("--- 开始 ThunderSVM SVR 最小化测试 ---")
#%%
try:
    # 1. 生成一份干净的、符合要求的回归数据
    #    使用 float32 类型，这在 GPU 计算中更常见
    print("正在生成 make_regression 数据 (10000 样本, 100 特征)...")
    X, y = make_regression(
        n_samples=10000, 
        n_features=100, 
        noise=0.1, 
        random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    print("数据生成完毕。")

    # 2. 实例化 SVR 模型
    #    使用默认参数，包括默认的 cache_size
    print("正在实例化 thundersvm.SVR()...")
    clf = SVR(verbose=True)
    print("实例化成功。")

    # 3. 拟合模型
    print("正在调用 clf.fit(X, y)...")
    clf.fit(X, y)
    print("\n--- ✅ 测试成功 ---")
    print("thundersvm.SVR 在标准数据集上成功完成了拟合。")

except Exception as e:
    print("\n--- ❌ 测试失败 ---")
    print("在测试过程中捕获到异常:")
    print(f"异常类型: {type(e).__name__}")
    print(f"异常信息: {e}")
# %%
