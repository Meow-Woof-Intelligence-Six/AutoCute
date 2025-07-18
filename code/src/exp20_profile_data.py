#%%
from typing import final
from auto_config import project_dir
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
df158 = pd.read_pickle(project_dir / "temp/qlib_alpha158_ranked_with_stock_info.pkl")
target_cols=['收盘', '涨跌幅', '涨跌幅排名', '涨跌正负', '涨跌', '龙虎']

df = df158
# 先把 target_cols 从原表中“拿走”
other_cols = [c for c in df.columns if c not in target_cols]

# 再按“其余列 + target_cols”重新拼接
df = df[other_cols + target_cols]
#%%
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['item_id', 'timestamp']).reset_index(drop=True)

alpha_cols = df.columns[2:].difference(target_cols).tolist()
alpha_cols
#%%
print("\n--- 步骤1: 构建未来标签 (Label Generation) ---")
# 核心思想：用明天的目标值，作为今天的标签。这需要对每个股票分组进行操作。
label_cols = {}
for col in target_cols:
    if col in df.columns:
        # 为每个股票分组，然后将目标列上移一位
        df[f'label_{col}'] = df.groupby('item_id')[col].shift(-1)
        label_cols[col] = f'label_{col}'

print(f"已生成 {len(label_cols)} 个标签列: {list(label_cols.values())}")
print("查看标签生成结果 (注意最后一行因为没有未来数据，所以标签为NaN):\n")
df[['item_id', 'timestamp', '收盘', 'label_收盘']].tail()
#%%
# 异常值处理
# 先转换原始特征的分布，再构造新特征。





#%%
# 生成更多新的特征
# 先构造新的特征，再为每一个任务筛选重要特征。
# 1. timestamp 扩展

# 2. 衍生截面维度特征
# 界面成交量等


# 1. 异常值处理

# 2. 转换分布
# box-cox, Yeo-Johnson, log1p, sqrt, etc.

# 3. 交叉


# 4. 检验


# %%
# from ydata_profiling import ProfileReport
# profile = ProfileReport(df, tsmode=True, sortby="timestamp", title="Time-Series EDA", 
# explorative=True)

# profile.to_file(project_dir / "temp/profile_report_ts.html")
#%%
numeric_cols = df.select_dtypes(include='number').columns.tolist()
final_feature_cols = df.columns[2:].difference(list(label_cols.values())).difference([
    "prompt"
]).intersection(numeric_cols).tolist()



print(f"最终特征列: {final_feature_cols}")


# %%
print("\n--- 步骤4: 数据整理 (保留NaN值) ---")
df_processed = df.copy() 
print(f"数据行数: {len(df_processed)} (未删除任何行，保留NaN用于分析)")

print("\n--- 步骤4.5: 数据类型转换 (确保标签列为数值类型) ---")
for col in label_cols.values():
    if col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
print("标签列已强制转换为数值类型，非数值内容已转换为NaN。")

print("\n--- 步骤5: 划分训练集与验证集 (防止过拟合的关键) ---")
df_for_split = df_processed.dropna(subset=list(label_cols.values()))
split_date = df_for_split['timestamp'].quantile(0.8, interpolation='nearest')
train_df = df_for_split[df_for_split['timestamp'] < split_date]
valid_df = df_for_split[df_for_split['timestamp'] >= split_date]
print(f"数据集划分完成:")
print(f"训练集时间范围: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
print(f"验证集时间范围: {valid_df['timestamp'].min()} -> {valid_df['timestamp'].max()}")
print(f"训练集大小: {len(train_df)}, 验证集大小: {len(valid_df)}")

#%%
#%%

from custom_ag.pretty_tree import classification_and_draw
binary_labels_to_test = ['label_涨跌正负', 'label_涨跌', 'label_龙虎']
for y_learn in binary_labels_to_test:
    for split, df in zip(["train", "valid"], [train_df, valid_df]):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train = df[final_feature_cols].fillna(0)  # 填充NaN以便模型训练
        Y_train = df[y_learn].fillna(0)  # 填充NaN以便模型训练
        dt = classification_and_draw(X_train.values, Y_train.values, X_train.columns,
                                        X_train,
                                        class_names=["否", f"{y_learn}"],
                                        path=project_dir/f"temp/实验_{split}_{y_learn}",
                                        dummy_indicator="-Is-", replacement={
                    "<=0.5": "False",
                    ">0.5": "True"
                },
        )
        # feature importance 从大到小，字典，特征
        print(f"\n===== {split} 集上 '{y_learn}' 的决策树 =====")
        feat_import = {
            col: imp for col, imp in zip(X_train.columns, dt.feature_importances_)
        }
        feat_import = {k: v for k, v in sorted(feat_import.items(), key=lambda item: item[1], reverse=True)}
        print("特征重要性 (从大到小):")
        for feature, importance in feat_import.items():  # 只显示前10个特征
            print(f"{feature}: {importance:.4f}")
        # 保存字典为json
        import json
        with open(project_dir / f"temp/实验_{split}_{y_learn}_feature_importance.json", 'w', encoding='utf-8') as f:
            json.dump(feat_import, f, ensure_ascii=False, indent=4)
#%%

#%%
print("\n--- 步骤6: 稳健特征检验与可视化 (使用Plotly) ---")

def get_significant_features(df, label_col, feature_cols, alpha=0.05):
    p_values = {}
    for feature in feature_cols:
        temp_df = df[[feature, label_col]].dropna()
        group0 = temp_df[temp_df[label_col] == 0][feature]
        group1 = temp_df[temp_df[label_col] == 1][feature]
        if len(group0) > 1 and len(group1) > 1:
            _, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
            p_values[feature] = p_value
    p_values_series = pd.Series(p_values)
    return p_values_series[p_values_series < alpha].sort_values()

binary_labels_to_test = ['label_涨跌正负', 'label_涨跌', 'label_龙虎']

for label in binary_labels_to_test:
    if label not in df_processed.columns:
        continue

    print(f"\n===== 开始检验标签: {label} =====")
    
    sig_features_train = get_significant_features(train_df, label, final_feature_cols)
    print(f"训练集上发现 {len(sig_features_train)} 个显著特征。")
    
    sig_features_valid = get_significant_features(valid_df, label, final_feature_cols)
    print(f"验证集上发现 {len(sig_features_valid)} 个显著特征。")

    robust_features = sig_features_train.index.intersection(sig_features_valid.index)
    print(f"==> 发现 {len(robust_features)} 个在训练集和验证集上都显著的【稳健特征】。")
    
    if len(robust_features) > 0:
        print("Top 5 最稳健的特征 (基于训练集p-value排序):")
        print(sig_features_train.loc[robust_features].head())

        # --- [NEW] 使用Plotly进行可视化 ---
        most_robust_feature = robust_features[0]
        
        # 1. 准备用于绘图的DataFrame
        train_plot_df = train_df[[most_robust_feature, label]].copy().dropna()
        train_plot_df['数据集'] = '训练集'
        
        valid_plot_df = valid_df[[most_robust_feature, label]].copy().dropna()
        valid_plot_df['数据集'] = '验证集'
        
        plot_df = pd.concat([train_plot_df, valid_plot_df])
        plot_df[label] = plot_df[label].astype(str) # 将标签转为字符串类型，方便Plotly识别为分类

        # 2. 创建小提琴图
        fig = px.violin(
            plot_df,
            x=label,
            y=most_robust_feature,
            color=label,
            box=True,  # 在小提琴内部显示箱线图
            facet_col='数据集', # <<< 使用分面，将训练集和验证集并排展示，对比更清晰
            title=f"【稳健特征】'{most_robust_feature}' 在训练集与验证集上的分布对比",
            labels={
                most_robust_feature: f"特征 '{most_robust_feature}' 的值",
                label: f"标签 '{label}' 的值",
                '数据集': '数据集'
            }
        )
        
        # 3. 显示图表
        fig.show()

    else:
        print("未发现任何稳健特征。")

print("\n--- 特征工程与稳健性检验完成 ---")

# %%
# 运行较慢
import warnings
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
print("\n--- 步骤7: 模型化特征选择 (互信息与LGBM重要性) ---")
binary_labels_to_test = ['label_涨跌正负', 'label_涨跌', 'label_龙虎']
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 准备训练数据，填充NaN以便模型训练
X_train = train_df[final_feature_cols].fillna(0) # 使用0填充NaN，对于树模型是安全的

for label in binary_labels_to_test:
    if label not in train_df.columns:
        continue
    
    print(f"\n===== 开始为标签 '{label}' 进行特征选择 =====")
    y_train = train_df[label]

    # --- 方法1: 互信息 (Mutual Information) ---
    print("\n--- 方法1: 互信息 (Mutual Information) ---")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    
    print("Top 10 特征 (根据互信息):")
    print(mi_series.head(10))

    # --- 方法2: LightGBM 特征重要性 ---
    print("\n--- 方法2: LightGBM 特征重要性 ---")
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
    lgbm.fit(X_train, y_train)
    
    importance_series = pd.Series(lgbm.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    
    # 筛选出重要性 > 0 的特征
    important_features = importance_series[importance_series > 0]
    print(f"发现 {len(important_features)} 个重要性 > 0 的特征。")
    print("Top 10 特征 (根据LGBM重要性):")
    print(important_features.head(10))

    # 可选：可视化LGBM特征重要性
    fig = px.bar(
        important_features.head(20).sort_values(ascending=True),
        orientation='h',
        title=f"LGBM特征重要性 Top 20 (标签: {label})",
        labels={'value': '重要性得分', 'index': '特征'}
    )
    fig.show()

# %%
