# %% 精简版 AgSVMModel
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.common import space
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

MAX_SAMPLES = 20_000     # 超阈值时仅保留尾部 2w 条

class AgSVMModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self.regression_or_classification = self.problem_type in ["regression", "softclass"]
        self.use_thunder_svm = False
        self._scaler = None          # 保存 StandardScaler

    # ---------- 1. 预处理：编码 + 标准化 + 截断 ----------
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)

        # --- LabelEncoder ---
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X)

        # --- 截断 ---
        if X.shape[0] > MAX_SAMPLES:
            X = X.tail(MAX_SAMPLES)

        # --- 缺失值/无穷处理 ---
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], [1e9, -1e9])

        # --- 标准化 ---
        num_cols = X.select_dtypes(include=["int", "float"]).columns
        if is_train:
            self._scaler = StandardScaler().fit(X[num_cols])
        X[num_cols] = self._scaler.transform(X[num_cols])

        return X.to_numpy(np.float32)

    # ---------- 2. 参数 & 训练 ----------
    def _get_model_params(self, **kwargs):
        p = super()._get_model_params(**kwargs)
        if p.get("gamma") == "scale":
            p["gamma"] = 1.0 / (self.n_features * (self.X_var or 1.0))
        elif p.get("gamma") == "auto":
            p["gamma"] = 1.0 / self.n_features
        p.setdefault("probability", True)
        return p

    def _fit(self, X, y, num_gpus=None, **kwargs):
        from sklearn.svm import SVC, SVR

        X = self.preprocess(X, is_train=True)
        self.X_var = float(X.var())
        self.n_features = X.shape[1]

        if num_gpus and num_gpus > 0:
            try:
                from thundersvm import SVC as TSVC, SVR as TSVR
                model_cls = TSVR if self.regression_or_classification else TSVC
                self.use_thunder_svm = True
            except ImportError:
                model_cls = SVR if self.regression_or_classification else SVC
        else:
            model_cls = SVR if self.regression_or_classification else SVC

        self.model = model_cls(**self._get_model_params())
        self.model.fit(X, y)

    # ---------- 3. 存 / 取 ----------
    def save(self, path=None, verbose=True):
        if not self.use_thunder_svm:
            return super().save(path, verbose)
        path = Path(path or self.path)
        path.mkdir(parents=True, exist_ok=True)

        model_bak = self.model
        self.model = None
        save_pkl.save(path / self.model_file_name, self, verbose=verbose)
        self.model = model_bak

        self.model.save_to_file(str(path / "thundersvm.model"))
        return str(path)

    @classmethod
    def load(cls, path, reset_paths=True, verbose=True):
        obj = load_pkl.load(Path(path) / cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)
        if not obj.use_thunder_svm:
            return super().load(path, reset_paths, verbose)

        from thundersvm import SVC, SVR
        model_cls = SVR if obj.regression_or_classification else SVC
        obj.model = model_cls(**obj._get_model_params())
        obj.model.load_from_file(str(Path(path) / "thundersvm.model"))
        return obj

    # ---------- 4. 其它 ----------
    def _set_default_params(self):
        base = {"C": 1.0, "kernel": "rbf", "gamma": "scale",
                "coef0": 0.0, "shrinking": True, "tol": 1e-3,
                "probability": True}
        if self.regression_or_classification:
            base["epsilon"] = 0.1
        else:
            base["class_weight"] = "balanced"
        for k, v in base.items():
            self._set_default_param_value(k, v)

    def _get_default_searchspace(self):
        return {
            "C": space.Real(1e-4, 1e4, log=True),
            "gamma": space.Categorical(["scale", "auto"]),
            "kernel": space.Categorical(["linear", "poly", "rbf", "sigmoid"]),
        }