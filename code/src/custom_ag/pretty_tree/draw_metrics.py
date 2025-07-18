from matplotlib import pyplot as plt
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
def best_threshold(y_test, y_pred_prob, metric=f1_score):
    best_threshold_value, best_score = 0, 0
    # for threshold in sorted(y_pred_prob):
    # ordered_set = sorted(set(y_pred_prob))
    # ordered_set = set(y_pred_prob)
    ordered_set = y_pred_prob
    for threshold in ordered_set:
        # TODO 这里其实没有覆盖所有的决策可能性
        # y_pred = y_pred_prob>threshold
        y_pred = y_pred_prob>=threshold
        new_score = metric(y_test, y_pred)
        if new_score>=best_score:
            best_score = new_score
            best_threshold_value = threshold
    return best_threshold_value, best_score
# best_threshold_value, best_score = best_threshold(y_test, y_pred_prob, metric=precision_score)
# best_threshold_value, best_score = best_threshold(y_test, y_pred_prob, metric=balanced_accuracy_score)
# best_threshold_value, best_score = best_threshold(y_test, y_pred_prob, metric=matthews_corrcoef)
# best_threshold_value, best_score = best_threshold(y_test, y_pred_prob)
# best_threshold_value, best_score

def fast_evaluation(y_test, y_pred_prob, threshold=None, metric=balanced_accuracy_score,return_threshold=False):
    if threshold is None:
        threshold, _ = best_threshold(y_test, y_pred_prob, metric=metric)
    y_pred = y_pred_prob>=threshold
    d = dict(roc_auc=roc_auc_score(y_test, y_pred_prob), 
                accuracy=accuracy_score(y_test, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
                mcc=matthews_corrcoef(y_test, y_pred),
                f1=f1_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred),
                recall=recall_score(y_test, y_pred))
    if return_threshold:
        return d, threshold
    return d
    
# fast_evaluation(y_test, y_pred_prob, threshold=best_threshold_value)


def plot_auc(y_test, y_pred_prob, 
             curve_name=None, title="受试者工作特征曲线",
             xlabel="假正例率", ylabel="真正例率"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    if curve_name is None:
        curve_name = f"auc={roc_auc:.2f}"
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='orange', label=curve_name)
    ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return roc_auc, fig

def plot_pr(y_test, y_pred_prob, 
            curve_name=None, title="精确率-召回率曲线",
            xlabel="召回率", ylabel="精确率"):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    if curve_name is None:
        curve_name = f"auc={auc(recall, precision):.2f}"
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='orange', label=curve_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return auc(recall, precision), fig



