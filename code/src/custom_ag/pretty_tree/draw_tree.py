# 'os' module provides functions for interacting with the operating system 
import os
# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np
# 'Pandas' is used for data manipulation and analysis
import pandas as pd
# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
from matplotlib import pyplot as plt
# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns
# to suppress warnings
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import tree
from sklearn.tree import _tree
from sklearn.base import is_classifier  # 用于判断是回归树还是分类树
from dtreeviz.colors import adjust_colors  # 用于分类树颜色（色盲友好模式）
import seaborn as sns  # 用于回归树颜色
from matplotlib.colors import Normalize  # 用于标准化RGB数值
import graphviz  # 插入graphviz库
import os

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def regression_and_draw(X, y, X_names, df, path, dummy_indicator="(单选)", replacement={}):
    # 有bug
    dt = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)

    dt.fit(X, y)

    dot_data = tree.plot_tree(dt, feature_names=X_names, )

    dot_data = tree_to_dot(dt, y, X, X_names, df,
                           dummy_indicator=dummy_indicator, replacement=replacement)
    graph = graphviz.Source(dot_data)
    dot_data = dot_data.replace('helvetica', 'MicrosoftYaHei')
    graph = graphviz.Source(dot_data)
    graph.render(filename=path, format="svg")
    return dt

from sklearn.model_selection import GridSearchCV

def classification_and_draw(X, y, X_names, df, class_names, path, dummy_indicator="(单选)",
                            replacement={}, dt_params=None,
                            auto_search=False,
                            ):
    if auto_search:
        parameters = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': list(range(2, 10)),
            'class_weight': ['balanced', None],
        }
        dtree = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(dtree, parameters, scoring='roc_auc', cv=3, refit=True)
        grid_search.fit(X, y)
        print(f"Local search cv best score is {grid_search.best_score_} via param {grid_search.best_params_}")
        dt = grid_search.best_estimator_
    else:
        if dt_params is None:
            dt_params = dict(
                criterion='log_loss',
                max_depth=4,
                class_weight="balanced"
            )
        dt = DecisionTreeClassifier(**dt_params)
        dt.fit(X, y)
    draw_classification(dt, X, y, X_names, df, class_names, path, dummy_indicator, replacement)
    return dt


def draw_classification(dt, X, y, X_names, df, class_names, path, dummy_indicator="(单选)", replacement={}):
    dot_data = tree.export_graphviz(dt,
                                    feature_names=X_names,
                                    class_names=class_names,
                                    rounded=True,
                                    special_characters=True,
                                    filled=True)

    dot_data = tree_to_dot(dt, y, X, X_names, df, class_names=class_names,
                           dummy_indicator=dummy_indicator, replacement=replacement)
    graph = graphviz.Source(dot_data)
    dot_data = dot_data.replace('helvetica', 'MicrosoftYaHei')
    graph = graphviz.Source(dot_data)
    graph.render(filename=path, format="svg")
    return graph


def draw_importance(dt, X_names, topk=10, title="特征重要性", ylabel="纯度降值"):
    importances = pd.Series(dt.feature_importances_, index=X_names)
    fig, ax = plt.subplots()
    importances = importances.sort_values(ascending=False).head(topk)
    importances.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, importances


def get_yvec_xmat_vnames(xmat, target, df):
    yvec = df[target]

    # 将拥有n个不同数值的变量转换为n个0/1的变量，变量名字中有dummy_indicator作为标注
    # xmat = pd.get_dummies(df.loc[:, df.columns != target], prefix_sep = dummy_indicator)

    vnames = xmat.columns

    return yvec, xmat, vnames


# 得到更好看的树！

def get_categorical_dict(df):
    # store all the values of categorical value
    df_categorical = df.select_dtypes(include=['object', 'bool', 'category'])
    categorical_dict = {}
    for i in df_categorical.columns:
        # store in descending order
        categorical_dict[i] = sorted(list(set(df[i].astype('str'))))
    return categorical_dict


def tree_to_dot(tree, yvec, xmat, vnames, df, class_names=None,
                dummy_indicator="(单选)", replacement={}):
    """ 把树变成dot data,用于输入graphviz然后绘制
    
    参数
        tree: DecisionTree的输出
        target: 目标变量名字
        df: 表单

    输出
        graphvic_str: dot data
    """
    # get yvec, vnames and categorical_dict of the df
    # yvec, xmat, vnames = get_yvec_xmat_vnames(target, df)

    categorical_dict = get_categorical_dict(df)

    if is_classifier(tree):
        # 如果是分类树
        # classes should be in descending order
        # class_names = sorted(list(set(yvec)))
        # class_names = [f"健康", f"有{y_columns[y_learn]}"]
        return classification_tree_to_dot(tree, vnames, class_names, categorical_dict,
                                          dummy_indicator=dummy_indicator, replacement=replacement)
    else:
        return regression_tree_to_dot(tree, vnames, categorical_dict,
                                      dummy_indicator=dummy_indicator, replacement=replacement)


# not_word = "不选"
not_word = "not"


def replace(threshold_string, replacement, sep=", "):
    thresholds = threshold_string.split(sep)
    for i, k in enumerate(thresholds):
        if k in replacement:
            thresholds[i] = replacement[k]
        if k.startswith(not_word):
            key = k[len(not_word):]
            if key in replacement:
                thresholds[i] = not_word + replacement[key]
    return sep.join(thresholds)


def classification_tree_to_dot(tree, feature_names, class_names, categorical_dict, dummy_indicator="(单选)",
                               replacement={}):
    """ 把分类树转化成dot data

    参数
        tree: DecisionTreeClassifier的输出
        feature_names: vnames, 除去目标变量所有变量的名字
        class_names: 目标变量所有的分类
        categorical_dict: 储存所有名称及分类的字典

    输出
        graphvic_str: the dot data
    """
    tree_ = tree.tree_

    # store colors that distinguish discrete chunks of data
    if len(class_names) <= 10:
        # get the colorblind friendly colors
        color_palette = adjust_colors(None)['classes'][len(class_names)]
    else:
        color_palette = sns.color_palette("coolwarm", len(class_names)).as_hex()

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # initialize the dot data string
    graphvic_str = 'digraph Tree {node [shape=oval, penwidth=0.1, width=1, fontname=helvetica] ; edge [fontname=helvetica] ;'

    # print(graphvic_str)

    def recurse(node, depth, categorical_dict):
        # store the categorical_dict information of each side
        categorical_dict_L = categorical_dict.copy()
        categorical_dict_R = categorical_dict.copy()
        # non local statement of graphvic_str
        nonlocal graphvic_str
        # variable is not dummy by default
        is_dummy = False
        # get the threshold
        threshold = tree_.threshold[node]

        # get the feature name
        name = feature_name[node]
        # judge whether a feature is dummy or not by the indicator dummy_indicator
        if dummy_indicator in str(name) and name.split(dummy_indicator)[0] in list(categorical_dict.keys()):
            is_dummy = True
            # if the feature is dummy, the threshold is the value following name
            name, threshold = name.split(dummy_indicator)[0], name.split(dummy_indicator)[1]

        # get the data distribution of current node
        value = tree_.value[node][0]
        # get the total amount
        n_samples = tree_.n_node_samples[node]
        # calculate the weight
        weights = [i / sum(value) for i in value]
        # get the largest class
        class_name = class_names[np.argmax(value)]

        # pair the color and weight
        fillcolor_str = ""
        for i, j in enumerate(color_palette):
            fillcolor_str += j + ";" + str(weights[i]) + ":"
        fillcolor_str = '"' + fillcolor_str[:-1] + '"'

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # if the node is not a leaf
            graphvic_str += ('{} [style=wedged, label=<{}<br/>{}>, fillcolor =' + fillcolor_str + '] ;').format(node,
                                                                                                                n_samples,
                                                                                                                name)
            # print(('{} [style=wedged, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,name))
            if is_dummy:
                # if the feature is dummy and if its total categories > 5
                categorical_dict_L[name] = [str(i) for i in categorical_dict_L[name] if i != threshold]
                categorical_dict_R[name] = [str(threshold)]
                if len(categorical_dict[name]) > 5:
                    # only show one category on edge
                    # threshold_left = "not " + threshold
                    threshold_left = not_word + threshold
                    threshold_right = threshold
                else:
                    # if total categories <= 5, list all the categories on edge
                    threshold_left = ", ".join(categorical_dict_L[name])
                    threshold_right = threshold
            else:
                # if the feature is not dummy, then it is numerical
                threshold_left = "<=" + str(round(threshold, 3))
                threshold_right = ">" + str(round(threshold, 3))

            # 字符串特殊判断

            threshold_left = replace(threshold_left, replacement)
            threshold_right = replace(threshold_right, replacement)

            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,
                                                                                                     tree_.children_left[
                                                                                                         node],
                                                                                                     threshold_left)
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,
                                                                                                      tree_.children_right[
                                                                                                          node],
                                                                                                      threshold_right)
            # print(('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left))
            # print(('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right))

            recurse(tree_.children_left[node], depth + 1, categorical_dict_L)
            recurse(tree_.children_right[node], depth + 1, categorical_dict_R)
        else:
            # the node is a leaf
            graphvic_str += (
                        '{} [shape=box, style=striped, label=<{}<br/>{}>, fillcolor =' + fillcolor_str + '] ;').format(
                node, n_samples, class_name)
            # print(('{} [shape=box, style=striped, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,class_name))

    recurse(0, 1, categorical_dict)
    return graphvic_str + "}"


def regression_tree_to_dot(tree, feature_names, categorical_dict, dummy_indicator="(单选)", replacement={}):
    """ 把回归树转换成dot data

    参数
        tree: DecisionTreeClassifier的输出
        feature_names: vnames, 除去目标变量所有变量的名字
        categorical_dict: 储存所有名称及分类的字典

    输出
        graphvic_str: the dot data
    """
    # get the criterion of regression tree: mse or mae
    criterion = tree.get_params()['criterion']

    tree_ = tree.tree_

    value_list = tree_.value[:, 0][:, 0]

    # Normalize data to produce heatmap colors
    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=min(value_list), vmax=max(value_list))
    rgb_values = (cmap(norm(value_list)) * 255).astype(int)
    hex_values = ['#%02x%02x%02x' % (i[0], i[1], i[2]) for i in rgb_values]

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # initialize the dot data string
    graphvic_str = 'digraph Tree {node [shape=oval, width=1, color="black", fontname=helvetica] ;edge [fontname=helvetica] ;'

    # print(graphvic_str)

    def recurse(node, depth, categorical_dict):
        # store the categorical_dict information of each side
        categorical_dict_L = categorical_dict.copy()
        categorical_dict_R = categorical_dict.copy()
        # non local statement of graphvic_str
        nonlocal graphvic_str

        # variable is not dummy by default
        is_dummy = False
        # get the threshold
        threshold = tree_.threshold[node]

        # get the feature name
        name = feature_name[node]
        # judge whether a feature is dummy or not by the indicator dummy_indicator
        if dummy_indicator in str(name) and name.split(dummy_indicator)[0] in list(categorical_dict.keys()):
            is_dummy = True
            # if the feature is dummy, the threshold is the value following name
            name, threshold = name.split(dummy_indicator)[0], name.split(dummy_indicator)[1]

        # get the regression value
        value = round(tree_.value[node][0][0], 3)
        # get the impurity
        impurity = criterion + "=" + str(round(tree_.impurity[node], 3))
        # get the total amount
        n_samples = tree_.n_node_samples[node]

        # pair the color with node
        fillcolor_str = '"' + hex_values[node] + '"'

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # if the node is not a leaf
            graphvic_str += (
                        '{} [style="filled", label=<{}<br/>{}<br/>{}>, fillcolor =' + fillcolor_str + '] ;').format(
                node, n_samples, impurity, name)
            # print(('{} [style="filled", label=<{}<br/>{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,impurity,name))
            if is_dummy:
                # if the feature is dummy and if its total categories > 5
                categorical_dict_L[name] = [str(i) for i in categorical_dict_L[name] if i != threshold]
                categorical_dict_R[name] = [str(threshold)]

                if len(categorical_dict[name]) > 5:
                    # only show one category on edge
                    threshold_left = not_word + threshold
                    threshold_right = threshold
                else:
                    # if total categories <= 5, list all the categories on edge
                    threshold_left = ", ".join(categorical_dict_L[name])
                    threshold_right = threshold
            else:
                # if the feature is not dummy, then it is numerical
                threshold_left = "<=" + str(round(threshold, 3))
                threshold_right = ">" + str(round(threshold, 3))
            # 字符串特殊判断
            threshold_left = replace(threshold_left, replacement)
            threshold_right = replace(threshold_right, replacement)

            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,
                                                                                                     tree_.children_left[
                                                                                                         node],
                                                                                                     threshold_left)
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,
                                                                                                      tree_.children_right[
                                                                                                          node],
                                                                                                      threshold_right)
            # print(('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left))
            # print(('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right))

            recurse(tree_.children_left[node], depth + 1, categorical_dict_L)
            recurse(tree_.children_right[node], depth + 1, categorical_dict_R)
        else:
            # the node is a leaf
            graphvic_str += (
                        '{} [shape=box, style=filled, label=<{}<br/>{}<br/>{}>, fillcolor =' + fillcolor_str + '] ;').format(
                node, n_samples, impurity, value)
            # print(('{} [shape=box, style=filled, label=<{}<br/>{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,impurity,value))

    recurse(0, 1, categorical_dict)
    return graphvic_str + "}"
