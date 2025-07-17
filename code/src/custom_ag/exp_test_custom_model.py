# %%
# %load_ext autoreload
# %autoreload 2
#%%
from autogluon.tabular import TabularDataset

train_data = TabularDataset('/home/ye_canming/repos/novelties/ts/comp/stork_predict_hyh/alpha158_train_converted.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('/home/ye_canming/repos/novelties/ts/comp/stork_predict_hyh/alpha158_valid_converted.csv')  # another Pandas DataFrame
# label = '龙虎'  # specifies which column do we want to predict
label = '涨跌'  # specifies which column do we want to predict
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

train_data.head(5)

#%%
from autogluon.tabular import TabularPredictor
from svm import AgSVMModel
from autogluon.tabular.models.tabpfn.tabpfn_model import TabPFNModel 
# 配置包含SVM的超参数
hyperparameters = {
    AgSVMModel: [
        {}, 
        {
            'gamma': 'auto',
        }, 
        {
            'kernel': 'linear',
            'gamma': 'auto',
        }, 
        {
            'kernel': 'poly',
            'gamma': 'auto',
        },
        {
            'kernel': 'sigmoid',
            'gamma': 'auto',
        }
    ],
    # TabPFNModel: {},
    # AssertionError: Max allowed features for the model is 100
}

# 初始化预测器
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    tuning_data=test_data,
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
