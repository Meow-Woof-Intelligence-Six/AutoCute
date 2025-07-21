

from pathlib import Path
this_file = Path(__file__).resolve()
this_dir = this_file.parent
import sys
sys.path.append(str(this_dir))
from stage60_evaluator import *
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3:
        gt_path = sys.argv[1]
        pred_path = sys.argv[2]
        res = evaluate(ground_truth_path=gt_path, prediction_path=pred_path)
        print(res)
    else:
        print("\n未检测到命令行输入，将使用交互式输入：")
        gt_path = input("请输入真实结果文件路径（如 evaluate/result.csv ）: ").strip()
        pred_path = input("请输入预测结果文件路径（如 your_prediction.csv ）: ").strip()

        if gt_path == "":
            gt_path = "ref/result_428.csv"
        if pred_path == "":
            pred_path = "/home/ye_canming/repos/novelties/ts/comp/stork_predict_hyh/autoglum_results.csv"

        res = evaluate(ground_truth_path=gt_path, prediction_path=pred_path,verbose=False)
        # print(res)
        print(res["final_score"])