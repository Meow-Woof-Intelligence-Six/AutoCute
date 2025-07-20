#%% 完整版特征选择脚本
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import lightgbm as lgb
from collections import Counter

# ---------- 配置 ----------
warnings.filterwarnings('ignore')
TOP_N_FEATURES = 216
VETTING_THRESHOLD = 2
SUBSAMPLE_SIZE = 100_000          # 设为 None 用全量

try:
    from auto_config import project_dir
    OUTPUT_JSON_PATH = project_dir / "temp/stage2/feature_selection_finance_results_vetted.json"
except ImportError:
    # 兜底路径（本地调试）
    project_dir = Path.cwd()
    OUTPUT_JSON_PATH = Path("temp/stage2/feature_selection_finance__results_vetted.json")

OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------- 1. 读数 ----------
def create_mock_data():
    mock_file = Path("temp/lag158_with_timestamp_features.pkl")
    mock_file.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range('2023-01-01', periods=300)
    items = [f'SH{600000 + i:04d}' for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, items], names=['timestamp', 'item_id'])
    df = pd.DataFrame(np.random.randn(len(idx), 200), index=idx,
                      columns=[f'feature_{i:03d}' for i in range(200)]).reset_index()
    df['industry'] = [f'ind_{i%5}' for i in range(len(df))]
    df['board']    = [f'board_{i%2}' for i in range(len(df))]

    for i, c in enumerate(["收盘_shift","涨跌幅_shift","涨跌幅排名_shift",
                           "涨跌正负_shift","涨跌_shift","龙虎_shift"]):
        df[c] = np.random.rand(len(df))
    df.to_pickle(mock_file)
    return df

try:
    # df = pd.read_pickle(project_dir / "temp/lag158_with_timestamp_features.pkl")
    df = pd.read_pickle(project_dir / "temp/lag158_finance_with_timestamp_features.pkl")
    print("✅ 成功加载真实数据")
except (ImportError, FileNotFoundError):
    print("⚠️  使用模拟数据")
    df = create_mock_data()

# ---------- 2. 定义列 ----------
label_cols   = ["收盘_shift", "涨跌幅_shift", "涨跌幅排名_shift",
                "涨跌正负_shift", "涨跌_shift", "龙虎_shift"]
id_cols      = ['timestamp', 'item_id']
reg_labels   = label_cols[:3]
clf_labels   = label_cols[3:]

numeric_feature_cols = sorted(
    set(df.select_dtypes(include=np.number).columns) - set(label_cols) - set(id_cols)
)
categorical_feature_cols = sorted(
    set(df.columns) - set(numeric_feature_cols) - set(label_cols) - set(id_cols)
)
feature_cols = numeric_feature_cols      # 仅数值特征参与筛选

print(f"数值特征 {len(feature_cols)} | 类别特征 {len(categorical_feature_cols)}")

# ---------- 3. 训练/验证划分 ----------
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=label_cols)
split_date = df_clean['timestamp'].quantile(0.8, interpolation='nearest')
train_df = df_clean[df_clean['timestamp'] < split_date].copy()
valid_df = df_clean[df_clean['timestamp'] >= split_date].copy()

# ⭐ 分类标签转整数
for col in clf_labels:
    train_df[col] = (train_df[col] > 0).astype('int8')
    valid_df[col] = (valid_df[col] > 0).astype('int8')

print(f"训练集 {len(train_df)} 行 | 验证集 {len(valid_df)} 行")

# ---------- 4. 特征选择函数 ----------
def get_top_features(df, label, method, is_regression):
    X = df[feature_cols].fillna(0)
    y = df[label]

    if method == 'spearman':
        corr_p = {f: spearmanr(X[f], y, nan_policy='omit') for f in X.columns}
        corr_df = pd.DataFrame(corr_p, index=['corr', 'p']).T
        return corr_df[corr_df['p'] < 0.05].sort_values(
            'corr', key=abs, ascending=False).head(TOP_N_FEATURES).index.tolist()

    if method == 'mannwhitneyu':
        g0, g1 = X[y == 0], X[y == 1]
        if g0.empty or g1.empty:
            return []
        pvals = {f: mannwhitneyu(g0[f], g1[f], alternative='two-sided').pvalue
                 for f in X.columns}
        pser = pd.Series(pvals)
        return pser[pser < 0.05].sort_values().head(TOP_N_FEATURES).index.tolist()

    if method == 'mi':
        mi_fn = mutual_info_regression if is_regression else mutual_info_classif
        scores = pd.Series(mi_fn(X, y, random_state=42), index=X.columns)
        return scores.sort_values(ascending=False).head(TOP_N_FEATURES).index.tolist()

    if method == 'lgbm':
        params = dict(
            objective='regression_l1' if is_regression else 'binary',
            n_estimators=50, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=-1
        )
        model = (lgb.LGBMRegressor(**params) if is_regression
                 else lgb.LGBMClassifier(**params))
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        return imp[imp > 0].head(TOP_N_FEATURES).index.tolist()
    return []

# ---------- 5. 特征选择与一致性 ----------
all_results = {}
tasks = {'regression': reg_labels, 'classification': clf_labels}

train_s = (train_df.sample(SUBSAMPLE_SIZE, random_state=42)
           if SUBSAMPLE_SIZE else train_df)
valid_s = (valid_df.sample(SUBSAMPLE_SIZE, random_state=42)
           if SUBSAMPLE_SIZE else valid_df)

for task_type, labels in tasks.items():
    is_reg = task_type == 'regression'
    for label in labels:
        if label not in df_clean.columns:
            continue
        print(f"\n==== {label} ({task_type}) ====")
        all_results[label] = {'intermediate_results': {}, 'final_results': {}}

        consistent = {}
        methods = ['spearman', 'mi', 'lgbm'] if is_reg else ['mannwhitneyu', 'mi', 'lgbm']

        for m in methods:
            tr = train_df if m in {'spearman', 'mannwhitneyu'} else train_s
            val = valid_df if m in {'spearman', 'mannwhitneyu'} else valid_s

            tr_top = get_top_features(tr, label, m, is_reg)
            val_top = get_top_features(val, label, m, is_reg)
            intersect = list(set(tr_top) & set(val_top))
            consistent[m] = intersect
            print(f"  {m}: train={len(tr_top)}, val={len(val_top)}, both={len(intersect)}")

        votes = Counter()
        for feats in consistent.values():
            votes.update(feats)

        vetted = [f for f, c in votes.items() if c >= VETTING_THRESHOLD]
        vetted.sort(key=lambda x: (-votes[x], x))
        all_results[label]['intermediate_results']['consistent_features'] = consistent
        all_results[label]['intermediate_results']['feature_votes'] = dict(votes)
        all_results[label]['final_results']['vetted_features'] = vetted
        print(f"  加冕特征 {len(vetted)} 个（≥{VETTING_THRESHOLD} 票）")

# ---------- 6. 保存 ----------
all_results['categorical_features_to_keep'] = categorical_feature_cols
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print("\n✅ 全部完成！结果保存在：", OUTPUT_JSON_PATH)