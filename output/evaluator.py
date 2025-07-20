import pandas as pd
from typing import Dict, Any
from pathlib import Path
this_file = Path(__file__).resolve()
this_dir = this_file.parent
import sys
sys.path.append(str(this_dir))

def _calculate_metrics_for_category(gt_series: pd.Series, pred_series: pd.Series) -> Dict[str, Any]:
    """
    为单个类别（如“涨幅最大”或“涨幅最小”）计算所有指标。

    Args:
        gt_series (pd.Series): 包含真实股票代码的Series。
        pred_series (pd.Series): 包含预测股票代码的Series。

    Returns:
        Dict[str, Any]: 包含该类别所有分数的字典。
    """
    # 确保输入数据长度为10
    if len(gt_series) != 10 or len(pred_series) != 10:
        raise ValueError(f"输入数据长度必须为10，但真实数据为{len(gt_series)}，预测数据为{len(pred_series)}")

    # 1. F1 分数计算
    gt_set = set(gt_series)
    pred_set = set(pred_series)
    
    intersection_count = len(gt_set.intersection(pred_set))
    
    # 根据赛题定义，Precision和Recall的分母都是10
    precision = intersection_count / 10.0
    recall = intersection_count / 10.0
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    # 2. 斯皮尔曼等级相关性计算
    N = 10
    gt_rank_map = {stock_code: rank + 1 for rank, stock_code in enumerate(gt_series)}
    
    # 对于预测列表中未出现在真实列表中的股票，给予一个惩罚性排名
    # 常见的惩罚排名是 N+1，这里我们使用 11
    # penalty_rank = N + 1
    
    sum_d_squared = 0
    for pred_rank, pred_code in enumerate(pred_series, 1):
        actual_rank = gt_rank_map.get(pred_code, None)
        d = pred_rank - actual_rank if actual_rank is not None else 10 # 如果未找到真实排名，则使用惩罚分数
        sum_d_squared += d ** 2
        
    # Spearman公式
    denominator = N * (N**2 - 1)
    spearman_corr = 1 - (6 * sum_d_squared) / denominator
    # 计算命中和未命中的股票代码
    hit_stocks = list(gt_set.intersection(pred_set))
    missed_gt_stocks = list(gt_set - pred_set)  # 真实股票中预测遗漏的
    wrong_pred_stocks = list(pred_set - gt_set)  # 预测中错误的股票
    return {
        "f1_score": f1_score,
        "spearman_corr": spearman_corr, 
        "intersection_count": intersection_count,
        "precision": precision,
        "recall": recall,
        "hit_stocks": hit_stocks,
        "missed_gt_stocks": missed_gt_stocks,
        "wrong_pred_stocks": wrong_pred_stocks
    }

def evaluate(ground_truth_path: str, prediction_path: str, verbose=True) -> Dict[str, Any]:
    """
    主评测函数，读取文件并计算最终分数。

    Args:
        ground_truth_path (str): 真实结果文件的路径 (check.csv)。
        prediction_path (str): 选手提交结果文件的路径 (result.csv)。

    Returns:
        Dict[str, Any]: 包含所有详细分数和最终分数的字典。
    """
    try:
        gt_df = pd.read_csv(ground_truth_path, dtype=str)
        pred_df = pd.read_csv(prediction_path, dtype=str)
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        return {}

    # --- 为“涨幅最大”类别计算分数 ---
    up_metrics = _calculate_metrics_for_category(
        gt_df['涨幅最大股票代码'],
        pred_df['涨幅最大股票代码']
    )
    
    # --- 为“涨幅最小”类别计算分数 ---
    down_metrics = _calculate_metrics_for_category(
        gt_df['涨幅最小股票代码'],
        pred_df['涨幅最小股票代码']
    )
    
    # --- 计算最终得分 ---
    final_score = (0.2 * up_metrics['f1_score'] + 
                   0.2 * down_metrics['f1_score'] + 
                   0.3 * up_metrics['spearman_corr'] + 
                   0.3 * down_metrics['spearman_corr'])

    # --- 整理并打印报告 ---
    if verbose:
        print("=" * 50)
        print("      股票预测评测报告")
        print("=" * 50)
        
        print("\n--- 涨幅最大股票 (Top 10 Up) ---\n")
        print(f"  - 预测命中数: {up_metrics['intersection_count']} / 10")
        print(f"  - 精度 (Precision): {up_metrics['precision']:.4f}")
        print(f"  - 召回率 (Recall):    {up_metrics['recall']:.4f}")
        print(f"  - F1 分数:          {up_metrics['f1_score']:.4f}")
        print(f"  - 斯皮尔曼等级相关性: {up_metrics['spearman_corr']:.4f}")

        print("\n--- 涨幅最小股票 (Top 10 Down) ---\n")
        print(f"  - 预测命中数: {down_metrics['intersection_count']} / 10")
        print(f"  - 精度 (Precision): {down_metrics['precision']:.4f}")
        print(f"  - 召回率 (Recall):    {down_metrics['recall']:.4f}")
        print(f"  - F1 分数:          {down_metrics['f1_score']:.4f}")
        print(f"  - 斯皮尔曼等级相关性: {down_metrics['spearman_corr']:.4f}")

        print("\n" + "=" * 50)
        print("      最终得分计算")
        print("=" * 50, "\n")
        print(f"  Final Score = (0.2 * F1_up) + (0.2 * F1_down) + (0.3 * Rank_up) + (0.3 * Rank_down)")
        print(f"              = (0.2 * {up_metrics['f1_score']:.4f}) + (0.2 * {down_metrics['f1_score']:.4f}) + "
            f"(0.3 * {up_metrics['spearman_corr']:.4f}) + (0.3 * {down_metrics['spearman_corr']:.4f})")
        print(f"\n  最终得分: {final_score:.4f}")
        print("\n" + "=" * 50)
    
    # 返回一个包含所有分数的字典，方便程序化使用
    up_metrics = {"up_metrics"+k: v for k, v in up_metrics.items()}
    down_metrics = {"down_metrics"+k: v for k, v in down_metrics.items()}
    full_results = {
        "final_score": final_score, 
    } | up_metrics | down_metrics
    return full_results
from pathlib import Path
if __name__ == '__main__':
    # 调用评测函数
    path = Path("train_ag/src/predictions/")
    ag_list = list(map(lambda x: x.as_posix(), path.glob('**/*.csv')))
    # ag_list = []
    hehe_list = [
        "/home/ye_canming/repos/novelties/ts/comp/stork_predict_hyh/autoglum_results.csv"
    ]

    all_results = []
    for prediction in [
        # 'train_qlib/result.csv',
        # 'train_qlib/result copy.csv',
        # 'baseline/output/result.csv',
        # ''
    ]+ag_list+hehe_list:
        print(prediction)
        res = evaluate(ground_truth_path='evaluate/result.csv', prediction_path=prediction)
        res['path'] = prediction
        all_results.append(res)
    all_results_df = pd.DataFrame(all_results)
    # 排序
    all_results_df = all_results_df.sort_values(by='final_score', ascending=False)
    print(all_results_df)
    # 保存结果
    all_results_df.to_csv(this_dir/'scores.csv', index=False)
    