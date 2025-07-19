# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# AutoGluon 核心组件
from autogluon.core.models import AbstractModel
from autogluon.common import space
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, R_OBJECT, S_BOOL

# Scikit-learn 预处理组件
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

class AgSVMModel(AbstractModel):
    """
    支持向量机 (SVM) 模型 for AutoGluon.
    
    核心特性:
    - **智能预处理**: 借鉴官方 LinearModel，使用 ColumnTransformer 对数值、偏态和类别特征进行独立且健壮的处理。
    - **大规模数据优化**: 当训练数据超过阈值时，自动使用最新的数据子集进行训练，以保证效率。
    - **GPU 加速**: 可选通过 ThunderSVM 实现 GPU 加速。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessor: ColumnTransformer | None = None
        self._numeric_features: list | None = None
        self._skewed_features: list | None = None
        self._categorical_features: list | None = None
        
        self.use_thunder_svm = False
        # 数据量阈值，超过此值将使用 tail 进行训练
        self.subsample_threshold = 20000

    def _get_feature_types(self, X: pd.DataFrame):
        """
        辅助函数：从特征元数据中获取并分离不同类型的特征。
        借鉴官方 LinearModel 的实现。
        """
        # 识别类别特征
        self._categorical_features = self.feature_metadata.get_features(valid_raw_types=[R_CATEGORY, R_OBJECT, S_BOOL])
        
        # 识别数值特征
        continuous_features = self.feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT], invalid_special_types=[S_BOOL])
        
        # 从数值特征中分离出偏态特征
        self._skewed_features = []
        self._numeric_features = []
        skew_threshold = self.params.get("proc.skew_threshold", 0.99)
        for feature in continuous_features:
            if np.abs(X[feature].skew()) > skew_threshold:
                self._skewed_features.append(feature)
            else:
                self._numeric_features.append(feature)

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """
        数据预处理。
        使用 ColumnTransformer 对不同类型的特征进行独立的、健壮的处理。
        """
        X.columns = X.columns.astype(str)
        
        if is_train:
            self._get_feature_types(X)
            
            # 为标准数值特征创建一个处理管道
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # 为偏态数值特征创建一个处理管道
            skewed_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('quantile', QuantileTransformer(output_distribution="normal", random_state=self.params.get('random_state', 0)))
            ])

            # 为类别特征创建一个处理管道 (简单填充)
            # SVM 需要数值输入，后续会通过 LabelEncoder 处理，这里仅填充
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])

            # 创建 ColumnTransformer
            self._preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self._numeric_features),
                    ('skew', skewed_transformer, self._skewed_features),
                    ('cat', categorical_transformer, self._categorical_features)
                ],
                remainder='passthrough'
            )
            X_transformed = self._preprocessor.fit_transform(X)
        else:
            if self._preprocessor is None:
                raise RuntimeError("Preprocessor is not fitted.")
            X_transformed = self._preprocessor.transform(X)
        
        # 最终处理：填充所有剩余的 NaN 并转换为 numpy 数组
        # 使用 np.nan_to_num 确保没有 NaN/inf 值进入 SVM
        return np.nan_to_num(X_transformed, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)


    def _get_model_params(self) -> dict:
        params = super()._get_model_params()
        # 计算gamma, thundersvm 不支持 自动决定
        if "gamma" in params:
            if params["gamma"] == "scale":
                # 确保 self.X_var 和 self.n_features 已经计算
                if hasattr(self, 'X_var') and hasattr(self, 'n_features'):
                    X_var = self.X_var
                    params["gamma"] = 1.0 / (self.n_features * X_var) if X_var != 0 else 1.0
                else:
                    # 如果在 HPO 等场景下未计算，则使用一个安全的回退值
                    params["gamma"] = 'auto' 
            if params["gamma"] == "auto":
                 if hasattr(self, 'n_features'):
                     params["gamma"] = 1.0 / self.n_features
                 else:
                     # 如果在 HPO 等场景下未计算，则使用一个安全的回退值
                     params["gamma"] = 0.1
        return params

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ):
        # 1. 数据量阈值控制
        if len(X) > self.subsample_threshold:
            print(f"数据量 ({len(X)}) 超过阈值 ({self.subsample_threshold})，将使用最新的 {self.subsample_threshold} 条数据进行训练。")
            X = X.tail(self.subsample_threshold)
            y = y.tail(self.subsample_threshold)

        # 2. 根据问题类型选择SVM模型
        # 防止报错，在这里才import
        from sklearn.svm import SVC, SVR
        num_gpus = kwargs.get('num_gpus', 0)
        if num_gpus > 0:
            try:
                from thundersvm import SVC, SVR
                print("Using ThunderSVM for GPU acceleration.")
                self.use_thunder_svm = True
            except ImportError:
                print("ThunderSVM not available, using scikit-learn SVC instead.")

        is_regression = self.problem_type in ["regression", "softclass"]
        model_cls = SVR if is_regression else SVC

        # 3. 预处理数据
        X_processed = self.preprocess(X, is_train=True)
        self.X_var = X_processed.var()
        self.n_features = X_processed.shape[1]

        # 4. 获取参数并训练模型
        params = self._get_model_params()

        # 修正：从参数中移除预处理相关的键，避免传递给 SVM 模型构造函数
        preprocessing_keys = {'proc.skew_threshold'}
        model_params = {k: v for k, v in params.items() if k not in preprocessing_keys}

        self.model = model_cls(**model_params)
        self.model.fit(X_processed, y)

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        # 修正：对于非 ThunderSVM 模型，直接使用父类的 save 方法，它会正确地 pickle 整个对象（包括预处理器）
        if not self.use_thunder_svm:
            return super().save(path=path, verbose=verbose)

        # ThunderSVM 的保存逻辑
        if path is None:
            path = self.path
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # 保存 thundersvm 模型文件
        real_model_path = os.path.join(path, "thundersvm.model")
        if verbose:
            print(f"Saving ThunderSVM model to {real_model_path}")
        self.model.save_to_file(real_model_path)
        
        # 保存 AutoGluon 的模型框架 (不包含实际模型)
        _model = self.model
        self.model = None
        save_pkl.save(path=os.path.join(path, self.model_file_name), object=self, verbose=verbose)
        self.model = _model
        
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        # 修正：对于非 ThunderSVM 模型，直接使用父类的 load 方法
        # 我们通过检查 thundersvm.model 文件是否存在来判断模型类型
        real_model_path = os.path.join(path, "thundersvm.model")
        if not os.path.exists(real_model_path):
            return super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        # ThunderSVM 的加载逻辑
        model = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
        if reset_paths:
            model.set_contexts(path)

        from thundersvm import SVC, SVR
        is_regression = model.problem_type in ["regression", "softclass"]
        model_cls = SVR if is_regression else SVC
        
        params = model._get_model_params()
        # 加载时不需要概率，可以加速
        params['probability'] = False
        
        model_params = {k: v for k, v in params.items() if k != 'proc.skew_threshold'}
        model.model = model_cls(**model_params)
        
        if verbose:
            print(f"Loading ThunderSVM model from {real_model_path}")
        model.model.load_from_file(real_model_path)
        return model

    def _set_default_params(self):
        default_params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "shrinking": True,
            "tol": 1e-3,
            "probability": True,
            "random_state": 0,
            "proc.skew_threshold": 0.99, # 偏态数据处理阈值
        }
        is_regression = self.problem_type in ["regression", "softclass"]
        if is_regression:
            default_params["epsilon"] = 0.1
        else:
            default_params["class_weight"] = "balanced"
        
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=["int", "float", "category"],
            ignored_type_group_special=["text"],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_searchspace(self):
        spaces = {
            "C": space.Real(1e-3, 1e3, log=True),
            "gamma": space.Categorical("scale", "auto"),
            "kernel": space.Categorical("rbf", "poly", "sigmoid"),
        }
        return spaces
