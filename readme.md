# 代码说明

[前面的部分保持不变，更新Stage 1细节]

## Stage 1: 数据准备

### 数据来源

1. 赛方提供数据
   - train.csv: 训练集交易数据
   - test.csv: 测试集交易数据
   - 位置: data/train.csv, data/test.csv

2. 基于赛方数据推导的特征
   - Alpha158: Qlib库中的技术分析因子
   - Alpha360: Qlib库中的另一组技术指标
   - 位置: temp/qlib_alpha158_ranked.pkl, temp/qlib_alpha360_ranked.pkl

3. 额外爬取的补充数据
   - 股票基本信息：上市时间、行业分类等
   - 位置: data/additional/all_stock_info_df.joblib

### 处理流程 (stage11_add_stock_info.py)

1. 股票基本信息处理
   ```python
   # 加载并验证数据完整性
   all_stock_info_df = load_and_validate_stock_info()
   
   # 数据清洗
   - 转换上市时间为datetime格式
   - 移除可能导致数据泄露的字段（"最新"、"总市值"、"流通市值"）
   - 标准化列名为英文
   ```

2. Alpha因子数据处理
   ```python
   # 处理Alpha158和Alpha360因子
   - 合并股票基本信息
   - 计算上市年限特征
   - 保存处理后的因子数据
   ```

3. 输出文件
   - temp/all_stock_info_df_cleaned.pkl：清洗后的股票信息
   - temp/qlib_alpha158_ranked_with_stock_info.pkl：带基本信息的Alpha158因子
   - temp/qlib_alpha360_ranked_with_stock_info.pkl：带基本信息的Alpha360因子

### 数据质量控制

1. 数据验证
   - 检查必要字段的完整性
   - 验证数据类型的正确性
   - 监控数据合并的匹配率

2. 防止数据泄露
   - 移除包含未来信息的字段
   - 仅使用历史数据计算特征
   - 保持时间序列的连续性

3. 错误处理
   - 完整的异常捕获和日志记录
   - 文件存在性检查
   - 数据类型转换保护

### 注意事项

1. 依赖检查
   ```bash
   # 运行前检查必要文件
   - data/additional/all_stock_info_df.joblib
   - temp/qlib_alpha158_ranked.pkl
   - temp/qlib_alpha360_ranked.pkl
   ```

2. 环境要求
   ```bash
   # 安装必要的Python包
   pip install pandas-stubs types-joblib
   ```

3. 目录结构
   ```
   project/
   ├── data/
   │   ├── additional/           # 补充数据
   │   │   └── all_stock_info_df.joblib
   │   ├── train.csv            # 训练数据
   │   └── test.csv             # 测试数据
   ├── temp/
   │   ├── stage1/              # 第一阶段处理结果
   │   └── qlib_alpha*_ranked.pkl  # Alpha因子数据
   └── code/
       └── src/
           └── stage11_add_stock_info.py
   ```

[后面的部分保持不变]
