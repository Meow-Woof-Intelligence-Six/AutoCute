#%%
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import lightgbm as lgb
import warnings
from collections import Counter

#%%
# --- 配置 ---
warnings.filterwarnings('ignore')
# 定义要为每种方法选择的top特征数量
TOP_N_FEATURES = 216
# 定义“加冕”特征的最低票数（一个特征至少要被这么多方法选中）
VETTING_THRESHOLD = 2 
# [优化] 为计算密集型方法定义子样本大小。设为None则使用全量数据。
SUBSAMPLE_SIZE = 100000 
# 定义输出文件名
from auto_config import project_dir
OUTPUT_JSON_PATH = project_dir/"temp/stage2/feature_selection_results_vetted.json"
OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

#%%
# --- 1. 加载预处理好的数据 ---
# 为了代码可独立运行，我们先创建一个模拟的pkl文件
def create_mock_data():
    mock_file = Path("temp/lag158_with_timestamp_features.pkl")
    if not mock_file.parent.exists():
        mock_file.parent.mkdir()
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=300))
    items = [f'SH{600000 + i:04d}' for i in range(50)]
    mock_index = pd.MultiIndex.from_product([dates, items], names=['timestamp', 'item_id'])
    data = np.random.randn(len(mock_index), 200) 
    columns = [f'feature_{i:03d}' for i in range(200)]
    mock_df = pd.DataFrame(data, index=mock_index, columns=columns).reset_index()
    # [新增] 添加模拟的类别型特征
    mock_df['industry'] = [f"industry_{i % 5}" for i in range(len(mock_df))]
    mock_df['board'] = [f"board_{i % 2}" for i in range(len(mock_df))]
    
    for i in range(6):
        mock_df[f'label_{i}'] = np.random.rand(len(mock_df))
    mock_df.rename(columns={
        'label_0': '收盘_shift', 'label_1': '涨跌幅_shift', 'label_2': '涨跌幅排名_shift',
        'label_3': '涨跌正负_shift', 'label_4': '涨跌_shift', 'label_5': '龙虎_shift'
    }, inplace=True)
    mock_df.to_pickle(mock_file)
    print("模拟数据文件已创建。")
    return mock_df

try:
    from auto_config import project_dir
    df = pd.read_pickle(project_dir / "temp/lag158_with_timestamp_features.pkl")
    print("成功加载真实数据。")
except (ImportError, FileNotFoundError):
    print("无法加载真实数据，将使用模拟数据。")
    df = create_mock_data()

#%%
# --- 2. 定义标签和特征列 ---
label_cols = ["收盘_shift", "涨跌幅_shift", "涨跌幅排名_shift", "涨跌正负_shift", "涨跌_shift", "龙虎_shift"]
id_cols = ['timestamp', 'item_id']
regression_labels = label_cols[0:3]
classification_labels = label_cols[3:6]

# 识别数值型特征
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_feature_cols = sorted(list(set(numeric_cols) - set(label_cols) - set(id_cols)))
print(f"识别出 {len(numeric_feature_cols)} 个【数值型】特征列进行筛选。")

# [新增] 识别类别型特征
all_cols = df.columns.tolist()
categorical_feature_cols = sorted(list(set(all_cols) - set(numeric_cols) - set(label_cols) - set(id_cols)))
print(f"识别出 {len(categorical_feature_cols)} 个【类别型】特征列将被保留: {categorical_feature_cols}")

# 更新 feature_cols 为仅数值型特征，用于后续计算
feature_cols = numeric_feature_cols

#%%
# --- 3. 划分训练集与验证集 ---
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=label_cols)
split_date = df_clean['timestamp'].quantile(0.8, interpolation='nearest')
train_df = df_clean[df_clean['timestamp'] < split_date]
valid_df = df_clean[df_clean['timestamp'] >= split_date]
print(f"训练集: {len(train_df)} 行, 验证集: {len(valid_df)} 行。")

#%%
# --- 4. 特征选择辅助函数 ---
def get_top_features(df, label, method, is_regression):
    # 注意：这里的 feature_cols 已经是纯数值型特征
    X = df[feature_cols].fillna(0)
    y = df[label]
    if method == 'spearman':
        corrs = {feat: spearmanr(X[feat], y, nan_policy='omit') for feat in X.columns}
        corr_df = pd.DataFrame(corrs, index=['corr', 'p']).T
        significant_features = corr_df[corr_df['p'] < 0.05]
        return significant_features.sort_values('corr', key=abs, ascending=False).head(TOP_N_FEATURES).index.tolist()
    
    if method == 'mannwhitneyu':
        group0 = X[y == 0]
        group1 = X[y == 1]
        if len(group0) < 1 or len(group1) < 1: return []
        p_values = {feat: mannwhitneyu(group0[feat], group1[feat], alternative='two-sided').pvalue for feat in X.columns}
        p_series = pd.Series(p_values)
        significant_features = p_series[p_series < 0.05]
        return significant_features.sort_values().head(TOP_N_FEATURES).index.tolist()

    if method == 'mi':
        mi_func = mutual_info_regression if is_regression else mutual_info_classif
        scores = pd.Series(mi_func(X, y, random_state=42), index=X.columns).sort_values(ascending=False)
        return scores.head(TOP_N_FEATURES).index.tolist()
    
    if method == 'lgbm':
        lgbm_params = {
            'objective': 'regression_l1' if is_regression else 'binary',
            'n_estimators': 50, 'learning_rate': 0.05, 'num_leaves': 31,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1
        }
        model = lgb.LGBMRegressor(**lgbm_params) if is_regression else lgb.LGBMClassifier(**lgbm_params)
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        return imp[imp > 0].head(TOP_N_FEATURES).index.tolist()
    return []

#%%
# --- 5. 执行特征选择与一致性检验 ---
all_results = {}
tasks = {'regression': regression_labels, 'classification': classification_labels}

train_sample = train_df.sample(n=min(len(train_df), SUBSAMPLE_SIZE), random_state=42) if SUBSAMPLE_SIZE else train_df
valid_sample = valid_df.sample(n=min(len(valid_df), SUBSAMPLE_SIZE), random_state=42) if SUBSAMPLE_SIZE else valid_df
print(f"为MI和LGBM创建子样本: 训练集样本 {len(train_sample)} 行, 验证集样本 {len(valid_sample)} 行。")

for task_type, labels in tasks.items():
    is_regression = task_type == 'regression'
    for label in labels:
        if label not in df_clean.columns: continue
        print(f"\n===== 开始处理任务: {label} ({task_type}) =====")
        all_results[label] = {'intermediate_results': {}, 'final_results': {}}
        
        consistent_features = {}
        methods = ['spearman', 'mi', 'lgbm'] if is_regression else ['mannwhitneyu', 'mi', 'lgbm']
        
        for method in methods:
            print(f"  - 方法: {method}, 进行Train/Val一致性检验...")
            train_data_to_use = train_df if method in ['spearman', 'mannwhitneyu'] else train_sample
            valid_data_to_use = valid_df if method in ['spearman', 'mannwhitneyu'] else valid_sample
            
            train_top = get_top_features(train_data_to_use, label, method, is_regression)
            valid_top = get_top_features(valid_data_to_use, label, method, is_regression)
            
            intersection = list(set(train_top) & set(valid_top))
            consistent_features[method] = intersection
            print(f"    - Train Top {len(train_top)}, Valid Top {len(valid_top)}, 一致特征: {len(intersection)}")
        
        all_results[label]['intermediate_results']['consistent_features'] = consistent_features

        print(f"  - 进行跨方法共识检验...")
        feature_votes = Counter()
        for method, features in consistent_features.items():
            feature_votes.update(features)
        
        all_results[label]['intermediate_results']['feature_votes'] = dict(feature_votes)

        vetted_features = [feat for feat, count in feature_votes.items() if count >= VETTING_THRESHOLD]
        vetted_features.sort(key=lambda x: (-feature_votes[x], x))

        all_results[label]['final_results']['vetted_features'] = vetted_features
        print(f"  - 最终加冕特征 ({len(vetted_features)}个, 票数 >= {VETTING_THRESHOLD}): {vetted_features[:10]}...")

#%%
# --- 6. 导出结果到JSON文件 ---
# [新增] 将识别出的类别型特征也加入到最终结果中
all_results['categorical_features_to_keep'] = categorical_feature_cols

with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"\n--- 特征筛选与检验完成 ---")
print(f"结果已成功导出到: {OUTPUT_JSON_PATH}")
print("\nJSON文件内容预览:")
print(json.dumps(all_results, indent=4, ensure_ascii=False)[:1500] + "\n...")
