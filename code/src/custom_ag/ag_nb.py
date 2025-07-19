import numpy as np
import pandas as pd
import warnings

# 确保 wnb 已经安装: pip install wnb
try:
    from wnb import GeneralNB, Distribution as D
except ImportError:
    raise ImportError("请先安装 'wnb' 库: pip install wnb")

# AutoGluon 核心类与特征生成器
from autogluon.core.models import AbstractModel
from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, R_OBJECT, S_BOOL, S_TEXT_AS_CATEGORY
from autogluon.features.generators import AsTypeFeatureGenerator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class IntelligentNaiveBayesModel(AbstractModel):
    """
    智能朴素贝叶斯模型 (IntelligentNaiveBayesModel) for AutoGluon.

    本模型利用了 `wnb` 库的 `GeneralNB` 分类器，并结合了 AutoGluon
    强大的特征类型推断能力。它会自动为不同类型的特征选择最合适的概率分布，
    从而构建一个比标准朴素贝叶斯更精确、更智能的分类器。

    自动分布映射逻辑:
    - **类别特征 (Categorical)**: 自动映射到 `GeneralNB` 的 `distributions` 参数，并指定为 `Distribution.CATEGORICAL`。
    - **数值特征 (Numeric)**: 根据 `wnb` 的默认行为，自动使用高斯分布 (正态分布)。
    - **文本特征 (Text)**: (待扩展) 未来可以集成 TF-IDF 等转换器。

    该模型目前仅支持分类任务。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_metadata_in = None
        self._numeric_features = []
        self._categorical_features = []
        # 使用更强大的 ColumnTransformer 作为预处理器
        self._preprocessor: ColumnTransformer | None = None

    def _get_feature_types(self, X: pd.DataFrame):
        """
        辅助函数：从特征元数据中获取数值和类别特征列表。
        修正：使用更鲁棒的方式识别特征类型。
        """
        self._feature_metadata_in = self.feature_metadata
        
        # 修正：通过查询“特殊类型”来识别类别特征，这是最可靠的方法。
        self._categorical_features = self._feature_metadata_in.get_features_by_special_types('category')
        
        # 数值特征是所有特征中排除了类别特征的部分。
        all_features = list(X.columns)
        self._numeric_features = [f for f in all_features if f not in self._categorical_features]

        # 确保特征列表中的列名确实存在于DataFrame中
        self._numeric_features = [f for f in self._numeric_features if f in X.columns]
        self._categorical_features = [f for f in self._categorical_features if f in X.columns]


    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """
        数据预处理。
        使用 ColumnTransformer 对不同类型的特征进行独立的、健壮的处理。
        """
        # 确保所有列名都是字符串，以避免 ColumnTransformer 报错
        X.columns = X.columns.astype(str)
        
        if is_train:
            self._get_feature_types(X)
            
            # 为数值特征创建一个处理管道
            n_quantiles = max(min(len(X) // 10, 1000), 10)
            random_state = self.params.get('random_state', 0)
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('quantile', QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=random_state))
            ])

            # 为类别特征创建一个处理管道
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32))
            ])

            # 创建 ColumnTransformer
            self._preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self._numeric_features),
                    ('cat', categorical_transformer, self._categorical_features)
                ],
                remainder='passthrough' # 保留其他列（如果有的话）
            )
            return self._preprocessor.fit_transform(X)
        else:
            if self._preprocessor is None:
                raise RuntimeError("Preprocessor is not fitted. Call `_preprocess` with `is_train=True` before calling it with `is_train=False`.")
            return self._preprocessor.transform(X)


    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             **kwargs):
        """
        模型训练的核心方法。
        此方法将分析特征类型，配置并训练 GeneralNB 模型。
        """
        if self.problem_type not in ['binary', 'multiclass']:
            raise ValueError("朴素贝叶斯模型目前仅支持 'binary' 或 'multiclass' 分类任务。")

        # 步骤 1: 预处理数据
        X_processed = self._preprocess(X, is_train=True)
        
        # 步骤 2: 获取类别特征在新数据中的索引
        # ColumnTransformer 会按照 transformers 列表的顺序排列输出列
        num_numeric_features = len(self._numeric_features)
        categorical_features_indices = list(range(num_numeric_features, num_numeric_features + len(self._categorical_features)))

        print(f"[IntelligentNaiveBayesModel] 自动特征类型识别完成:")
        print(f"  - 数值特征 ({len(self._numeric_features)}): {self._numeric_features[:5]}...")
        print(f"  - 类别特征 ({len(self._categorical_features)}): {self._categorical_features[:5]}...")

        # 步骤 3: 实例化并配置 GeneralNB 模型
        params = self._get_model_params()
        
        # 仅当存在类别特征时，才构造并传递 distributions 参数
        if categorical_features_indices:
            distributions = [(D.CATEGORICAL, categorical_features_indices)]
            self.model = GeneralNB(distributions=distributions, **params)
        else:
            self.model = GeneralNB(**params)
        
        # 步骤 4: 训练模型
        self.model.fit(X_processed, y)

    def _predict_proba(self, X, **kwargs):
        """
        概率预测方法。
        """
        X_processed = self._preprocess(X, **kwargs)
        return self.model.predict_proba(X_processed)

    def _set_default_params(self):
        """
        为 GeneralNB 模型设置默认超参数。
        """
        default_params = {
            'alpha': 1.0, # 拉普拉斯平滑系数 (for categorical features)
            'var_smoothing': 1e-9, # 方差平滑项 (for Gaussian features)
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        """
        定义模型可以处理的原始数据类型。
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category', 'object'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def get_info(self) -> dict:
        info = super().get_info()
        info['numeric_features_identified'] = self._numeric_features
        info['categorical_features_identified'] = self._categorical_features
        return info
