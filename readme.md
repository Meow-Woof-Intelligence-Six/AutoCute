# 代码说明

## 环境配置

- Python 3.11
- AutoGluon 1.3.1 (基础镜像: autogluon/autogluon:1.3.1-cuda12.4-jupyter-ubuntu22.04-py3.11)
- 主要依赖：
  - autogluon==1.3.1
  - pandas
  - feature-engine
  - flash-attn
  - akshare
  - ydata-profiling
  - qlib (使用本地修改版: ./third_parties/qlib)

完整依赖见requirements.txt，部分依赖库需要编译安装。

## 数据

1. 赛方提供数据
   - train.csv: 训练集交易数据
   - test.csv: 测试集交易数据

2. 基于赛方数据推导的特征
   - Alpha158因子：Qlib库中的技术分析因子
   - Alpha360因子：Qlib库中的另一组技术指标
   - 时间特征：基于交易日期生成的周期性特征

3. 额外爬取的补充数据
   - 股票基本信息：上市时间、行业分类等（存储于data/additional/all_stock_info_df.joblib）
   - 财报数据：用于补充基本面特征

## 预训练模型

1. TabPFN预训练模型
   - 路径：data/pretrained/tabpfn/*
   - 来源：PriorLabs的TabPFN项目
   - 用途：初始化TabPFNV2Model

2. Chronos预训练模型
   - 路径：data/pretrained/autogluon/chronos-bolt-*
   - 模型：chronos-bolt-base和chronos-bolt-small
   - 用途：用于时序预测的迁移学习

## 算法

### 整体思路介绍

按照代码中的stage序列进行处理：

1. Stage 1: 数据准备和基础处理
   - stage11_add_stock_info.py: 整合股票基本信息
   - 处理财报数据和其他补充信息

2. Stage 2: 特征工程和选择
   - 特征重要性评估
   - 特征筛选和验证
   - 保存特征选择结果(feature_selection_results_vetted.json)

3. Stage 3: 数据预处理
   - 数据类型转换和标准化
   - 缺失值处理
   - 划分训练集、验证集和测试集

4. Stage 4: 多目标模型训练
   - stage40_train_updown.py: 涨跌预测(分类问题)
   - stage42_train_price_change.py: 涨跌幅排名预测(回归问题)
   - stage43_train_price.py: 价格预测(回归问题)
   - stage44_train_price_agts_best_quality.py: 高质量价格预测

5. Stage 5: 结果生成和集成
   - stage54_make_results.py: 生成最终预测结果

### 网络结构

1. 时序预测模型集合
   - DeepAR: LSTM基础结构
   - Chronos: 预训练Transformer架构
   - AutoARIMA: 统计时间序列模型
   - AutoETS: 指数平滑模型
   - Theta/DynamicOptimizedTheta: 统计模型

2. 分类/回归模型集合
   - TabPFNV2: 基于预训练的表格数据模型
   - SVM: 支持GPU加速的改进实现
   - LightGBM/CatBoost: 梯度提升树模型
   - FASTAI: 深度学习模型

### 损失函数

根据不同任务采用不同损失函数：

1. 涨跌预测(stage40)
   - 评估指标：ROC-AUC
   - 损失函数：交叉熵损失

2. 涨跌幅排名预测(stage42)
   - 评估指标：Spearman相关系数
   - 损失函数：排序损失

3. 价格预测(stage43/44)
   - 评估指标：MASE和SMAPE
   - 损失函数：均方误差和对称平均绝对百分比误差

### 数据扩增

不使用传统的数据扩增方法，主要通过特征工程扩充信息：

1. 技术指标
   - Alpha158因子
   - Alpha360因子
   - 自定义技术指标

2. 时序特征
   - 交易日历特征
   - 周期性编码
   - 上市时间特征

### 模型集成

采用多层次的模型集成策略：

1. 单任务集成
   - AutoGluon的Weighted Ensemble
   - 交叉验证集成
   - Bagging集成

2. 多任务集成
   - 不同目标模型的预测结合
   - 基于验证集表现的动态权重

## 训练流程

1. 环境准备
```bash
/bin/bash /app/init.sh  # 安装依赖
```

2. 数据处理和特征工程
```python
# Stage 1: 添加股票基本信息
python code/src/stage11_add_stock_info.py

# Stage 2: 特征选择
python code/src/stage2/feature_selection.py

# Stage 3: 数据预处理
# 在各个训练脚本中自动执行
```

3. 多阶段训练
```python
# Stage 4: 模型训练
python code/src/stage40_train_updown.py  # 涨跌预测
python code/src/stage42_train_price_change.py  # 涨跌幅排名
python code/src/stage43_train_price.py  # 价格预测
python code/src/stage44_train_price_agts_best_quality.py  # 高质量价格预测

# Stage 5: 生成结果
python code/src/stage54_make_results.py
```

## 推理流程

1. 数据准备
   - 加载测试数据
   - 进行与训练时相同的特征工程
   - 准备模型输入格式

2. 模型预测
   - 加载各阶段训练好的模型
   - 执行多模型预测
   - 集成预测结果

3. 结果生成
```python
python code/src/stage54_make_results.py  # 生成最终提交文件
```

## 其他注意事项

1. 数据划分
   - 采用严格的时间序列划分
   - 验证集使用最近的5天数据
   - 测试集为最后一天数据
   - 使用auto_config.py自动推导日期范围

2. 性能优化
   - 使用OpenBLAS多线程优化(64线程)
   - GPU加速支持
   - 内存优化模式

3. 环境要求
   - CUDA 12.4
   - 32GB以上内存
   - 100GB以上磁盘空间
   - 支持多进程训练
