# %%
import os
import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.common import space

from autogluon.core.utils.loaders import load_json, load_pkl
from autogluon.core.utils.savers import save_json, save_pkl
from pathlib import Path

class AgSVMModel(AbstractModel):
    """
    支持向量机的优点
    - 高维有效（特征数量多，甚至比样本量多）
    - 推理内存少
    - 核函数
    缺点
    - 特征太多时，需要正则化
    - 默认不存在概率估计，需要昂贵的五折交叉验证才有概率估计。
    概率估计
    - 二分类，Platt scaling
    - 如果只是需要距离，不是需要概率，那么
    - probability=False
    - decision_function
    """

    def __init__(self, **kwargs):
        # super帮我们管理参数，具体来说是
        # path: str | None = None,
        # name: str | None = None,
        # problem_type: str | None = None,
        # eval_metric: str | metrics.Scorer | None = None,
        # hyperparameters: dict | None = None,
        super().__init__(**kwargs)
        # 额外存储的特征生成器
        self._feature_generator = None
        self.regression_or_classification = self.problem_type in [
            "regression",
            "softclass",
        ]
        self.use_thunder_svm = False

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:  # type: ignore
        # 父类数据预处理：处理缺失值和分类特征
        X = super()._preprocess(X, **kwargs)

        # 训练数据用来初始化特征生成器
        # LabelEncoderFeatureGenerator
        # 简单地把 文本、对象类别，变成整数编码
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        # 具体转换
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )

        # 填充缺失值为0（SVM不能处理NaN）
        # sklearn numpy比df快
        X = X.fillna(0)

        # X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py
        # for col in X.columns:
        #     # FIXME: Such behavior is very weird
        #     X[col] = X[col].replace([np.inf, -np.inf], X[col][~np.isinf(X[col])].mean())

        # df.replace(np.inf, np.finfo(np.float32).max, inplace=True)
        X.replace(np.inf, 1e9, inplace=True)
        X.replace(-np.inf, -1e9, inplace=True)
        return X.to_numpy(dtype=np.float32)

    def _get_model_params(self, convert_search_spaces_to_default: bool = False) -> dict:
        params = super()._get_model_params(convert_search_spaces_to_default=convert_search_spaces_to_default)
        # 计算gamma, thundersvm 不支持 自动决定
        if "gamma" in params:
            print(f"Number of features: {self.n_features}")
            if params["gamma"] == "scale":
                # var = E[X^2] - E[X]^2 if sparse
                X_var = self.X_var
                params["gamma"] = 1.0 / (self.n_features * X_var) if X_var != 0 else 1.0
                print("X variance:", X_var)
                print("self.n_features * X_var:", self.n_features * X_var)
                print("X_var != 0 ", X_var != 0)


            elif params["gamma"] == "auto":
                params["gamma"] = 1.0 / self.n_features
            print(f"Using gamma: {params['gamma']:.4e}")
            assert params["gamma"] >= 0, "gamma must be positive or 'scale' or 'auto'"
            if params["gamma"] == 0:
                print("Warning: gamma is set to 0, which may lead to poor performance. Consider using 'scale' or 'auto'.")

        # 设置概率输出用于后续集成
        if "probability" not in params:
            params["probability"] = True
        return params

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        X_unlabeled: pd.DataFrame = None,
        time_limit: float = None,
        sample_weight: pd.Series = None,
        sample_weight_val: pd.Series = None,
        num_cpus: int = None,
        num_gpus: int = None,
        verbosity: int = 2,
        **kwargs,
    ):
        # 拟合事件：至少O(N^2)，N为样本量。一万以上样本就慢。
        # 根据问题类型选择SVM模型
        # 防止报错，在这里才import
        from sklearn.svm import SVC, SVR

        if num_gpus is not None and num_gpus > 0:
            try:
                from thundersvm import SVC, SVR

                print("Using ThunderSVM for GPU acceleration.")
                self.use_thunder_svm = True
            except ImportError:
                from sklearn.svm import SVC

                print("ThunderSVM not available, using scikit-learn SVC instead.")

        if self.regression_or_classification:
            model_cls = SVR
        else:
            model_cls = SVC

        # 需要再fit里面做 preprocess
        X = self.preprocess(X, is_train=True)
        self.X_var = X.var()
        self.n_features = X.shape[1]


        # 当前我们需要的参数是什么
        params = self._get_model_params()

        self.model = model_cls(**params)
        self.model.fit(X, y)

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        if not self.use_thunder_svm:
            return super().save(path=path, verbose=verbose)
        if path is None:
            path = self.path

        Path(path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(path, self.model_file_name)

        # 保存表面的模型文件
        # 暂时隔离
        _model = self.model
        self.model = None
        # 自我保存
        save_pkl.save(path=file_path, object=self, verbose=verbose)
        # 恢复模型
        self.model = _model  

        # 保存真正的模型文件
        real_model_path =  os.path.join(path, "thundersvm.model")
        # https://github.com/Xtra-Computing/thundersvm/tree/master/python
        if verbose:
            print(f"Saving ThunderSVM model to {real_model_path}")
        self.model.save_to_file(real_model_path)
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        file_path = os.path.join(path, cls.model_file_name)
        model = load_pkl.load(path=file_path, verbose=verbose)
        if reset_paths:
            model.set_contexts(path)

        if not model.use_thunder_svm:
            return super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        

        from thundersvm import SVC, SVR
        if model.regression_or_classification:
            model_cls = SVR
        else:
            model_cls = SVC

        params = model._get_model_params()

        model.model = model_cls(**params)

        real_model_path =  os.path.join(path, "thundersvm.model")
        if verbose:
            print(f"Loading ThunderSVM model from {real_model_path}")
        model.model.load_from_file(real_model_path)
        return model

    def _set_default_params(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        # 默认超参数
        default_params = {
            "C": 1.0,  # 减少C可以增加正则化
            "kernel": "rbf",
            "gamma": "scale",
            "coef0": 0.0,  # 多项式核函数的常数项
            "shrinking": True,  # 启用收缩启发式
            "tol": 1e-3,  # 收敛容忍度
            "degree": 3,  # 多项式核函数的度数
            "random_state": 0,
            "probability": True,  # 启用概率预测用于集成
        }
        if self.regression_or_classification:
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
            task_default_params = {
                "epsilon": 0.1,  # 回归模型的松弛变量
            }
        else:
            task_default_params = {
                "class_weight": "balanced",
                # "break_ties": False,
            }
        default_params |= task_default_params
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        # 定义模型支持的输入类型
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=["int", "float", "category"],
            ignored_type_group_special=["text"],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_searchspace(self):
        # 定义HPO搜索空间

        spaces = {
            "C": space.Real(1e-4, 1e4, log=True),
            "gamma": space.Categorical(["scale", "auto"]),
            "kernel": space.Categorical(["linear", "poly", "rbf", "sigmoid"]),
        }
        if self.regression_or_classification:
            task_spaces = dict()
        else:
            task_spaces = dict()
        spaces |= task_spaces
        return spaces
