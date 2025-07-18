import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch


def ensure_pd_1d(series):
    if isinstance(series, torch.Tensor):
        series = series.cpu().detach().numpy()
    return pd.Series(series)


import seaborn as sns


def see_continuous_dist(one_d_feature):
    sns.scatterplot(one_d_feature)
    plt.show()
    sns.histplot(one_d_feature, bins=30, kde=True)
    plt.show()


from scipy.stats import shapiro, kstest
import numpy as np
import statsmodels.api as sm


def test_continuous_is_gaussian(one_d_feature):
    # 绘制QQ图
    # sm.qqplot(transformed_data, line='45', fit=True)
    sm.qqplot(one_d_feature, line='45', fit=True)
    # 添加标题
    plt.title("QQ Plot for Normal Distribution")
    # 显示图形
    plt.show()

    # 进行Shapiro-Wilk测试
    statistic, p_value = shapiro(one_d_feature)

    # 打印测试结果
    print("Shapiro-Wilk Test Statistic:", statistic)
    print("p-value:", p_value)

    # 根据p-value进行判断
    alpha = 0.05  # 显著性水平
    if p_value > alpha:
        print("数据可能符合正态分布")
    else:
        print("数据不符合正态分布")

    # 进行Kolmogorov-Smirnov测试
    statistic_ks, p_value_ks = kstest(one_d_feature, 'norm')

    # 打印测试结果
    print("Kolmogorov-Smirnov Test Statistic:", statistic_ks)
    print("p-value:", p_value_ks)

    # 根据p-value进行判断
    alpha = 0.05  # 显著性水平
    if p_value_ks > alpha:
        print("数据可能符合正态分布")
    else:
        print("数据不符合正态分布")

    return [(statistic, p_value), (statistic_ks, p_value_ks)]


#
# def see_discrete_dist(df, col_name):
#     print(df[col_name].value_counts())
#     plt.figure(figsize=(10, 5))
#     plt.title(f"{col_name} distribution")
#     plt.bar(df[col_name].value_counts().index, df[col_name].value_counts().values)
def see_discrete_dist(df, col_name, class_names=None, figsize=(10, 5), rotation=45,
                      title_name=None, color='skyblue', font_size=12):
    value_counts = df[col_name].value_counts()
    print(value_counts)

    if class_names is None:
        class_names = value_counts.index.tolist()

    plt.figure(figsize=figsize)
    plt.title(f"{col_name} Distribution" if title_name is None else title_name, fontsize=font_size)
    bars = plt.bar(class_names, value_counts.values, color=color, edgecolor='black')

    # 添加具体的 y 值标注
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())),
                 ha='center', va='bottom', fontsize=font_size)

    # plt.xlabel(col_name, fontsize=font_size)
    plt.ylabel("Count", fontsize=font_size)
    plt.xticks(rotation=rotation, fontsize=font_size)
    # plt.legend([f"{col_name} Count"], fontsize=font_size)
    # plt.show()


# def draw_discrete(df_col):
#     if isinstance(df_col, )
#     df_col.value_counts().plot(kind='pie')
# value = np.array(df_col)
# np.valu
# plt.pie(
# plt.axis('equal')  # 显示为圆（避免比例压缩为椭圆）
# plt.show()

if __name__ == '__main__':
    a = [1, 2, 3, 3]
    # a = np.array(a)
    a = torch.tensor(a).cuda()
    a = pd.Series(a)
    a.value_counts().plot(kind='pie')
    plt.show()
