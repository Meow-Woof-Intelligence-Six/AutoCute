#%%
import pandas as pd
from auto_config import project_dir
df158 = pd.read_pickle(project_dir / "temp/lag158_with_timestamp_features.pkl")
df158
# %%
df = df158
label_cols = [
    "收盘_shift",
    "涨跌幅_shift",
    "涨跌幅排名_shift",
    "涨跌正负_shift",
    "涨跌_shift",
    "龙虎_shift",
]
regression_labels_to_test = label_cols[0:3]
binary_labels_to_test = label_cols[3:6]

numeric_cols = df.select_dtypes(include='number').columns.tolist()
final_feature_cols = df.columns[2:].difference(label_cols).difference([
    "prompt"
]).intersection(numeric_cols).tolist()
final_feature_cols
# %%
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
import plotly.express as px
import warnings
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import lightgbm as lgb

df_for_split = df.dropna(subset=label_cols)
split_date = df_for_split['timestamp'].quantile(0.95, interpolation='nearest')
train_df = df_for_split[df_for_split['timestamp'] < split_date]
valid_df = df_for_split[df_for_split['timestamp'] >= split_date]
print(f"数据集划分完成:")
print(f"训练集时间范围: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
print(f"验证集时间范围: {valid_df['timestamp'].min()} -> {valid_df['timestamp'].max()}")
print(f"训练集大小: {len(train_df)}, 验证集大小: {len(valid_df)}")
#%%
# 准备训练数据，填充NaN以便模型训练
X_train = train_df[final_feature_cols].fillna(0) 

for label in regression_labels_to_test:
    if label not in train_df.columns:
        continue
    
    print(f"\n\n===== 开始为回归标签 '{label}' 进行特征选择 =====")
    y_train = train_df[label]

    # --- 方法1: 斯皮尔曼相关性及显著性检验 (Spearman Correlation with p-value) ---
    print("\n--- 方法1: 斯皮尔曼相关性及显著性检验 (Spearman Correlation with p-value) ---")
    
    correlations = {}
    p_values = {}
    
    # 逐个计算特征与目标的相关性和p值
    for feature in X_train.columns:
        # dropna() 确保只使用特征和目标都存在的行进行计算
        temp_df = pd.concat([X_train[feature], y_train], axis=1).dropna()
        if temp_df.shape[0] > 1:
            corr, p_val = spearmanr(temp_df[feature], temp_df[label])
            correlations[feature] = corr
            p_values[feature] = p_val

    # 整理成DataFrame方便分析
    corr_df = pd.DataFrame({
        'correlation': pd.Series(correlations),
        'p_value': pd.Series(p_values)
    }).dropna()
    
    # 筛选出统计上显著的特征 (p-value < 0.05)
    significant_features_df = corr_df[corr_df['p_value'] < 0.05].copy()
    significant_features_df['abs_correlation'] = significant_features_df['correlation'].abs()
    
    # 按相关性绝对值对显著特征进行排序
    significant_features_df = significant_features_df.sort_values(by='abs_correlation', ascending=False)
    
    print(f"发现 {len(significant_features_df)} 个与 '{label}' 显著相关的特征 (p < 0.05)。")
    if not significant_features_df.empty:
        print("Top 10 最显著相关的特征:")
        print(significant_features_df[['correlation', 'p_value']].head(10))
    else:
        print("未发现任何统计上显著相关的特征。")


    # # --- 方法2: 回归互信息 (Mutual Information for Regression) ---
    # print("\n--- 方法2: 回归互信息 (Mutual Information) ---")
    # mi_scores_reg = mutual_info_regression(X_train, y_train, random_state=42)
    # mi_series_reg = pd.Series(mi_scores_reg, index=X_train.columns).sort_values(ascending=False)
    
    # print("Top 10 特征 (根据回归互信息):")
    # print(mi_series_reg.head(10))

    # # --- 方法3: LightGBM 回归器特征重要性 ---
    # print("\n--- 方法3: LightGBM 回归器特征重要性 ---")
    # lgbm_reg = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=-1)
    # lgbm_reg.fit(X_train, y_train)
    
    # importance_series_reg = pd.Series(lgbm_reg.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    
    # important_features_reg = importance_series_reg[importance_series_reg > 0]
    # print(f"发现 {len(important_features_reg)} 个重要性 > 0 的特征。")
    # print("Top 10 特征 (根据LGBM回归器重要性):")
    # print(important_features_reg.head(10))
    
    # --- 可视化最佳特征与回归目标的关系 ---
    if not significant_features_df.empty:
        top_feature = significant_features_df.index[0]
        fig = px.scatter(
            train_df, 
            x=top_feature, 
            y=label, 
            trendline="ols", # 添加普通最小二乘趋势线
            trendline_color_override="red",
            title=f"'{top_feature}' 与回归目标 '{label}' 的关系 (训练集)",
            labels={'x': f"特征 '{top_feature}' 的值", 'y': f"目标 '{label}' 的值"}
        )
        fig.show()


print("\n--- 所有特征选择方法执行完毕 ---")
# %%
