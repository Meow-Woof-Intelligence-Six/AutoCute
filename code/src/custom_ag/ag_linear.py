import numpy as np
import pandas as pd
import warnings
from scipy.sparse import issparse

# AutoGluon 核心类与特征生成器
from autogluon.core.models import AbstractModel
from autogluon.common.features.types import R_INT, R_FLOAT
from autogluon.features.generators import LabelEncoderFeatureGenerator

# scikit-learn 模型与预处理工具
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV, ElasticNetCV,
    LogisticRegressionCV, RidgeClassifierCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV


class IntelligentLinearModel(AbstractModel):
    """
    智能线性模型 (IntelligentLinearModel) for AutoGluon.

    本模型将您关于 scikit-learn 线性模型的深度研究集成到了一个统一的接口中。
    它能够自动处理回归和分类任务。在 `_fit` 阶段，它会分析输入数据的特性，
    并根据您设计的决策流程图，自动选择最合适的线性模型进行训练。

    决策逻辑核心 (回归):
    1.  **高维**: LassoCV (L1正则化)
    2.  **多重共线性**: RidgeCV (L2正则化)
    3.  **稀疏特征**: ElasticNetCV (L1/L2混合)
    4.  **常规**: 普通最小二乘 (OLS)

    决策逻辑核心 (分类):
    1.  **高维**: LogisticRegressionCV (L1正则化)
    2.  **多重共线性**: RidgeClassifierCV (经校准以输出概率)
    3.  **稀疏特征**: LogisticRegressionCV (ElasticNet正则化)
    4.  **常规**: LogisticRegressionCV (L2正则化)

    所有正则化模型均使用其交叉验证版本 (CV)，以自动寻找最佳的正则化强度。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scaler = None  # 特征缩放器
        self._imputer = None # 缺失值填充器
        self._feature_generator = None # 类别特征编码器
        self._selected_model_name = None # 记录内部最终选择的模型名称
        self._is_classification = None # 标记是否为分类任务

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """
        数据预处理方法。
        该方法负责处理类别特征、填充缺失值，并进行特征标准化。
        这是所有线性模型取得良好性能的关键步骤。
        """
        # 1. 拷贝数据，避免修改原始 DataFrame
        X = X.copy()

        # 2. 类别特征处理
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator and self._feature_generator.features_in:
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        
        # 将所有数据转换为数值类型
        X = X.to_numpy(dtype=np.float32)

        # 3. 缺失值处理
        if is_train:
            self._imputer = SimpleImputer(strategy='mean')
            self._imputer.fit(X)
        if self._imputer:
            X = self._imputer.transform(X)

        # 4. 特征标准化 (Standardization)
        if is_train:
            self._scaler = StandardScaler()
            self._scaler.fit(X)
        if self._scaler:
            X = self._scaler.transform(X)

        return X

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             **kwargs):
        """
        模型训练的核心方法。
        此方法将执行您设计的决策流程，自动选择并训练模型。
        """
        # 步骤 0: 确定任务类型并预处理数据
        self._is_classification = self.problem_type in ['binary', 'multiclass']
        n_samples, n_features = X.shape
        X_processed = self.preprocess(X, is_train=True)
        
        # 初始化诊断信息
        condition_number = np.nan
        sparsity = np.nan

        # 步骤 1: 根据任务类型（回归 vs 分类）选择决策分支
        if not self._is_classification:
            # --- 回归任务决策逻辑 ---
            if n_features > n_samples:
                self._selected_model_name = 'LassoCV'
                alphas = np.logspace(-4, 0, 30)
                self.model = LassoCV(alphas=alphas, cv=5, random_state=self.random_state, n_jobs=-1)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    corr_matrix = np.corrcoef(X_processed, rowvar=False)
                    if np.isfinite(corr_matrix).all():
                        condition_number = np.linalg.cond(corr_matrix)
                    else:
                        condition_number = 1
                
                if condition_number > 30:
                    self._selected_model_name = 'RidgeCV'
                    alphas = np.logspace(-3, 3, 100)
                    self.model = RidgeCV(alphas=alphas, cv=5)
                else:
                    sparsity = np.count_nonzero(X_processed == 0) / X_processed.size
                    if sparsity > 0.6:
                        self._selected_model_name = 'ElasticNetCV'
                        l1_ratios = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
                        alphas = np.logspace(-4, 0, 30)
                        self.model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=5, random_state=self.random_state, n_jobs=-1)
                    else:
                        self._selected_model_name = 'LinearRegression'
                        self.model = LinearRegression(n_jobs=-1)
        else:
            # --- 分类任务决策逻辑 ---
            if n_features > n_samples:
                self._selected_model_name = 'LogisticRegressionCV(L1)'
                self.model = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='saga', random_state=self.random_state, n_jobs=-1, max_iter=1000)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    corr_matrix = np.corrcoef(X_processed, rowvar=False)
                    if np.isfinite(corr_matrix).all():
                        condition_number = np.linalg.cond(corr_matrix)
                    else:
                        condition_number = 1
                
                if condition_number > 30:
                    # 初始选择 RidgeClassifierCV
                    base_model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 100), cv=5)
                    with warnings.catch_warnings():
                        # 忽略预期的病态矩阵警告，因为这正是我们选择岭回归的原因
                        warnings.simplefilter("ignore", category=UserWarning)
                        base_model.fit(X_processed, y)
                    
                    # 使用 CalibratedClassifierCV 包装以获得概率预测
                    self.model = CalibratedClassifierCV(base_model, cv="prefit", method='isotonic')
                    self._selected_model_name = 'RidgeClassifierCV_Calibrated'
                else:
                    sparsity = np.count_nonzero(X_processed == 0) / X_processed.size
                    if sparsity > 0.6:
                        self._selected_model_name = 'LogisticRegressionCV(ElasticNet)'
                        self.model = LogisticRegressionCV(Cs=10, cv=5, penalty='elasticnet', solver='saga', l1_ratios=[0.1, 0.5, 0.9], random_state=self.random_state, n_jobs=-1, max_iter=1000)
                    else:
                        self._selected_model_name = 'LogisticRegressionCV(L2)'
                        self.model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='saga', random_state=self.random_state, n_jobs=-1, max_iter=1000)
        
        # 步骤 2: 训练选定的模型
        # 对于 RidgeClassifierCV, base_model 已被拟合, CalibratedClassifierCV 也需要被拟合
        self.model.fit(X_processed, y)
        
        # 打印出最终选择的模型，以便调试和验证
        print(f"[IntelligentLinearModel] 任务类型: {'分类' if self._is_classification else '回归'}")
        print(f"[IntelligentLinearModel] 数据特征分析: n_samples={n_samples}, n_features={n_features}, "
              f"collinearity_cond={condition_number:.2f}, sparsity={sparsity:.2f}")
        print(f"[IntelligentLinearModel] 根据决策流程，最终选择的模型为: {self._selected_model_name}")

    def _set_default_params(self):
        pass

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def get_info(self) -> dict:
        info = super().get_info()
        info['selected_model_name'] = self._selected_model_name
        return info
