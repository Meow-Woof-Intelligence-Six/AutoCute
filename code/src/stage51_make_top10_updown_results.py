#%%
import pandas as pd
from auto_config import project_dir

#%% --- 1. 定义合并和处理结果的函数 ---
def process_and_save_results(top10_proba_path, updown_proba_path, output_path, topk=10):
    import pandas as pd
    from auto_config import project_dir

    # 读取 top10_proba 和 updown_proba
    top10_proba = pd.read_csv(top10_proba_path)
    updown_proba = pd.read_csv(updown_proba_path)

    # 合并 top10_proba 和 updown_proba 并重命名列
    top10_updown_merged = top10_proba.merge(
        updown_proba,
        on=['item_id', 'timestamp'],
        how='outer'  # 或 'left' / 'inner' 按需选择
    )
    rename_map = {
        'False_x': 'False_top10',
        'True_x':  'True_top10',
        'False_y': 'False_updown',
        'True_y':  'True_updown'
    }
    top10_updown_merged = top10_updown_merged.rename(columns=rename_map)

    # 构造表格并排序
    # 1) 计算阈值
    avg = top10_updown_merged[['True_updown', 'False_updown']].mean(axis=1)

    # 2) 选 Top10 的 item_id（升序）
    top10 = (
        top10_updown_merged
        .sort_values('True_top10', ascending=True)
        .head(topk)['item_id']
        .tolist()
    )

    # 3) 选最小 10 的 item_id（升序后整体倒序）
    bottom10 = (
        top10_updown_merged
        .sort_values('True_top10', ascending=True)
        .head(topk)['item_id']
        .tolist()[::-1]  # 倒序
    )

    # 4) 按条件分组
    mask = top10_updown_merged['True_updown'] > avg

    # 5) 构造结果
    results = pd.DataFrame({
        '涨幅最大股票代码': top10 if mask.any() else None,
        '涨幅最小股票代码': bottom10 if not mask.all() else None
    })

    # 只保留非空列
    results = results.dropna(how='all')

    # 保存结果
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[✅] results 已保存至 {output_path}")

#%% --- 2. 第一次运行：使用 test1 数据 ---
process_and_save_results(
    top10_proba_path=project_dir / "temp/stage4/top10_test1_proba.csv",
    updown_proba_path=project_dir / "temp/stage4/updown_test1_proba.csv",
    output_path=project_dir / "temp/output/results_top10_updown_test1.csv"
)

#%% --- 3. 第二次运行：使用 test 数据 ---
process_and_save_results(
    top10_proba_path=project_dir / "temp/stage4/top10_test_proba.csv",
    updown_proba_path=project_dir / "temp/stage4/updown_test_proba.csv",
    output_path=project_dir / "temp/output/results_top10_updown_test.csv"
)


# top10_proba = pd.read_csv(project_dir / "temp/stage4/top10_test1_proba.csv")
# updown_proba = pd.read_csv(project_dir / "temp/stage4/updown_test1_proba.csv")
# topk = 10
# # top10_updown_merged = top10_proba.merge(updown_proba, on='item_id',"timestamp", how='left')
# # %%
# # 合并 top10_proba 和 updown_proba并重命名列
# top10_updown_merged  = top10_proba.merge(
#     updown_proba,
#     on=['item_id', 'timestamp'],
#     how='outer'   # 或 'left' / 'inner' 按需选择
# )
# rename_map = {
#     'False_x': 'False_top10',
#     'True_x':  'True_top10',
#     'False_y': 'False_updown',
#     'True_y':  'True_updown'
# }

# top10_updown_merged = top10_updown_merged.rename(columns=rename_map)


# # %%
# # 构造表格并排序
# # 1) 计算阈值
# avg = top10_updown_merged[['True_updown', 'False_updown']].mean(axis=1)

# # 2) 选 Top10 的 item_id（升序）
# top10 = (
#     top10_updown_merged
#     .sort_values('True_top10', ascending=True)
#     .head(topk)['item_id']
#     .tolist()
# )

# # 3) 选最小 10 的 item_id（升序后整体倒序）
# bottom10 = (
#     top10_updown_merged
#     .sort_values('True_top10', ascending=True)
#     .head(topk)['item_id']
#     .tolist()[::-1]        # 倒序
# )

# # 4) 按条件分组
# mask = top10_updown_merged['True_updown'] > avg

# # 5) 构造结果
# results = pd.DataFrame({
#     '涨幅最大股票代码': top10    if mask.any() else None,
#     '涨幅最小股票代码': bottom10 if not mask.all() else None
# })

# # 只保留非空列
# results = results.dropna(how='all')

# # %%
# # 假设 results 是 DataFrame，包含你的 top10 updown 排名结果

# save_path = project_dir / "output/results_top10_updown.csv"

# # 保存为 utf-8-sig 避免中文乱码，带列名 & 不带索引
# results.to_csv(save_path, index=False, encoding="utf-8-sig")

# print(f"[✅] results 已保存至 {save_path}")


# # %%
