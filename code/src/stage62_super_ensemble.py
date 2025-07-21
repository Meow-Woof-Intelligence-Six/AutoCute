from pathlib import Path
import shutil
from stage60_evaluator import evaluate  # 确保可直接 import

# ==== 配置路径 ====
from auto_config import project_dir

gt_path = project_dir / "temp/ref/test_set1_competition.csv"
output_dir = project_dir / "temp/output"
result_dir = project_dir / "output"
result_dir.mkdir(parents=True, exist_ok=True)

# ==== 遍历评分 ====
best_score = -100
best_file = None
scores_dict = {}

for csv_path in output_dir.glob("*1.csv"):  # **只遍历带 1 的文件**
    try:
        res = evaluate(ground_truth_path=gt_path, prediction_path=csv_path, verbose=False)
        score = res["final_score"]
        scores_dict[csv_path.name] = score
        print(f"{csv_path.name}: {score:.6f}")

        if score > best_score:
            best_score = score
            best_file = csv_path

    except Exception as e:
        print(f"[ERROR] {csv_path} 评分失败：{e}")

# ==== 找到对应的去掉 1 的文件并复制 ====
if best_file is not None:
    best_name = best_file.stem  # 如 "good1"
    best_suffix = best_file.suffix  # ".csv"

    if best_name.endswith("1"):
        stripped_name = best_name[:-1]  # "good"
        stripped_file_path = best_file.parent / f"{stripped_name}{best_suffix}"

        if stripped_file_path.exists():
            target_path = result_dir / "result.csv"
            shutil.copy(stripped_file_path, target_path)
            print(f"\n✅ 成功将 {stripped_file_path.name} 复制到 {target_path}")
        else:
            print(f"\n❌ 未找到对应的 {stripped_file_path.name}，请检查是否存在以便复制。")
    else:
        print("\n❌ 理论上应仅评分带 '1' 的文件，但当前最佳文件不带 '1'，请检查逻辑。")
else:
    print("❌ 未找到可用文件或评分均失败。")

