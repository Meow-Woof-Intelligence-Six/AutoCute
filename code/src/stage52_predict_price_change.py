#%%
TARGET_LABEL = '涨跌幅排名_shift'
from stage31_get_vetted_data import MODEL_OUTPUT_BASE_PATH
import pandas as pd

predictions = pd.read_csv(MODEL_OUTPUT_BASE_PATH / f"{TARGET_LABEL}_test.csv")
predictions1 = pd.read_csv(MODEL_OUTPUT_BASE_PATH / f"{TARGET_LABEL}_test1.csv")


#%%
# --- 4. 进行预测 ---
print("\n--- [4/5] 进行预测 ---")
def make_predictions_and_save_results(predictions, output_path):    
    # 创建结果数据框
    sorted_df = predictions.sort_values(by="prediction", ascending=True)
    top_10_item_ids = sorted_df["item_id"].head(10).astype(int).tolist()
    # 提取涨幅最小的后10个item_id
    bottom_10_item_ids = sorted_df["item_id"].tail(10).astype(int).tolist()
    
    # --- 创建新的DataFrame ---
    final_result_df = pd.DataFrame(
        {"涨幅最大股票代码": top_10_item_ids, "涨幅最小股票代码": bottom_10_item_ids}
    )
    final_result_df.to_csv(output_path, index=False)
    
    print(f"结果已保存至: {output_path}")
    return final_result_df

# 调用函数
from auto_config import project_dir
make_predictions_and_save_results(
    predictions = predictions, 
    output_path = project_dir / f"temp/output/results_{TARGET_LABEL}_test.csv"
)

make_predictions_and_save_results(
    predictions = predictions1, 
    output_path = project_dir / f"temp/output/results_{TARGET_LABEL}_test1.csv"
)

# %%
