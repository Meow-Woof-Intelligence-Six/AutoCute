from __future__ import annotations

import os
import numpy as np
import pandas as pd
import warnings

# AutoGluon 核心组件
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.core.utils import generate_train_test_split
from autogluon.features.generators import LabelEncoderFeatureGenerator

class TabPFNModel(AbstractModel):
    """
    AutoGluon model wrapper for the TabPFN v2.0 model: https://github.com/PriorLabs/TabPFN

    This wrapper is designed for TabPFN v2.0 and includes support for both classification and regression.
    TabPFN is a powerful pre-trained Transformer that excels on small tabular problems (typically < 10,000 rows).

    **Key Limitations (v2.0)**:
    - **Data Size**: Optimized for datasets up to 10,000 rows.
    - **Feature Count**: Limited to a maximum of 100 features.
    - **Class Count (Classification)**: Limited to a maximum of 10 classes.
    - **Preprocessing**: The model performs best with raw, unscaled numerical data and simple integer-encoded categoricals.
    """
    ag_key = "TABPFN"
    ag_name = "TabPFN"
    ag_priority = 110

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        # Custom file name for the fitted TabPFN model
        self.tabpfn_model_file = "fitted_tabpfn_model.pt"

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_gpus: int = 0, **kwargs):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.model.loading import save_fitted_tabpfn_model
        except ImportError:
            raise ImportError('tabpfn v2.0+ is not installed. Please run "pip install -U tabpfn"')
        
        # 1. Select model class based on problem type
        is_regression = self.problem_type == REGRESSION
        model_cls = TabPFNRegressor if is_regression else TabPFNClassifier

        # 2. Validate constraints before any processing
        self._validate_constraints(X, y)
        
        # 3. Subsample data if necessary
        ag_params = self._get_ag_params()
        sample_rows = ag_params.get("sample_rows")
        if sample_rows is not None and len(X) > sample_rows:
            X, y = self._subsample_train(X=X, y=y, num_rows=sample_rows)

        # 4. Preprocess data (minimal encoding, no scaling/imputation)
        X_processed = self.preprocess(X, is_train=True)

        # 5. Determine device for training
        device = "cuda" if num_gpus > 0 else "cpu"
        # device = "cuda" 
        
        # 6. Get hyperparameters and train the model
        params = self._get_model_params()
        
        # , **params
        self.model = model_cls(device=device, ignore_pretraining_limits=True).fit(
            X_processed, y, 
            # overwrite_warning=True # Suppresses warnings about exceeding the 1024 row limit
        )

    def _validate_constraints(self, X: pd.DataFrame, y: pd.Series):
        """Check if the dataset meets TabPFN's constraints."""
        ag_params = self._get_ag_params()
        max_features = ag_params.get("max_features")
        max_classes = ag_params.get("max_classes")
        max_rows = 10000

        if len(X) > max_rows:
            warnings.warn(
                f"{self.name} is optimized for datasets with <= {max_rows} rows, but data has {len(X)} rows. "
                f"Performance may be suboptimal. Consider subsampling.",
                UserWarning
            )
        # if self.problem_type != REGRESSION and self.num_classes > max_classes:
        #     raise AssertionError(
        #         f"Max allowed classes for {self.name} is {max_classes}, but found {self.num_classes} classes."
        #     )
        # if len(X.columns) > max_features:
        #     raise AssertionError(
        #         f"Max allowed features for {self.name} is {max_features}, but found {len(X.columns)} features. "
        #         f"Please perform feature selection."
        #     )

    def _subsample_train(self, X: pd.DataFrame, y: pd.Series, num_rows: int, random_state=0) -> tuple[pd.DataFrame, pd.Series]:
        """Stratified subsampling to reduce the training set size."""
        num_rows_to_drop = len(X) - num_rows
        X, _, y, _ = generate_train_test_split(
            X=X, y=y, problem_type=self.problem_type, test_size=num_rows_to_drop, random_state=random_state, min_cls_count_train=1
        )
        return X, y

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """
        Prepares data for TabPFN v2.0.
        - Converts categorical features to integer codes.
        - Converts all data to float32 numpy array.
        - Does NOT scale or impute NaNs, as TabPFN v2.0 handles this internally.
        """
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator and self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        
        # TabPFN expects a float32 numpy array
        X = X.to_numpy(dtype=np.float32)
        return X

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """
        Saves the AutoGluon wrapper and the fitted TabPFN model separately.
        """
        if path is None:
            path = self.path
        
        # Import the save function from TabPFN
        from tabpfn.model.loading import save_fitted_tabpfn_model
        
        # Save the fitted TabPFN model using its dedicated function
        if self.model is not None:
            tabpfn_model_path = os.path.join(path, self.tabpfn_model_file)
            if verbose:
                print(f"Saving TabPFN model to {tabpfn_model_path}")
            save_fitted_tabpfn_model(self.model, tabpfn_model_path)

        # Save the AutoGluon wrapper object (without the fitted model)
        _model = self.model
        self.model = None
        save_path = super().save(path=path, verbose=verbose)
        self.model = _model
        
        return save_path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """
        Loads the AutoGluon wrapper and then loads the fitted TabPFN model.
        """
        # Import the load function from TabPFN
        from tabpfn.model.loading import load_fitted_tabpfn_model
        
        # Load the AutoGluon wrapper object
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        
        # Load the fitted TabPFN model using its dedicated function
        tabpfn_model_path = os.path.join(path, model.tabpfn_model_file)
        if verbose:
            print(f"Loading TabPFN model from {tabpfn_model_path}")
        
        # Determine device; prefer CPU for loading to avoid memory issues on different hardware
        device = "cpu"
        
        model.model = load_fitted_tabpfn_model(tabpfn_model_path, device=device)
        return model

    def _set_default_params(self):
        """TabPFN v2.0 is largely hyperparameter-free."""
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        # TabPFN v2.0 supports both classification and regression.
        return [BINARY, MULTICLASS, REGRESSION]

    def _get_default_auxiliary_params(self) -> dict:
        """
        Default settings for TabPFN.
        - `sample_rows`: Performance stagnates around 4000 rows, so we subsample to this by default.
        - `max_features`: Hard limit of 100 features.
        - `max_classes`: Hard limit of 10 classes for classification.
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({
            "sample_rows": 4096,
            "max_features": 100,
            "max_classes": 10,
        })
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        TabPFN is a single, powerful model. Bagging it provides little benefit and is extremely slow.
        Therefore, we disable bagging by default.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update({
            "max_sets": 1,
            "fold_fitting_strategy": "sequential_local",
        })
        return default_ag_args_ensemble

    def _ag_params(self) -> set:
        return {"sample_rows", "max_features", "max_classes"}

    def _more_tags(self) -> dict:
        """
        Because TabPFN doesn't use validation data, it supports refit_full natively.
        """
        return {"can_refit_full": True}
