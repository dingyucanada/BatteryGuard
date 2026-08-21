"""B2 boosted-tree model with an explicit CPU-only sklearn fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from batteryguard.prediction.baselines import (
    ModelNotFittedError,
    numeric_feature_frame,
    validate_target,
)


class XGBoostLifetimeModel:
    """XGBoost-compatible wrapper that remains runnable without xgboost.

    ``model_name`` and ``used_fallback`` always disclose which implementation
    actually fitted the data; no report can silently call the fallback XGBoost.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 250,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        allow_fallback: bool = True,
        **extra_params: Any,
    ) -> None:
        if n_estimators < 1 or max_depth < 1 or learning_rate <= 0:
            raise ValueError("invalid boosted-tree hyperparameters")
        self.params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "random_state": int(random_state),
            **extra_params,
        }
        self.allow_fallback = bool(allow_fallback)
        self.feature_names_in_: tuple[str, ...] = ()
        self.estimator_: Any | None = None
        self.is_fitted_ = False
        self.used_fallback = False
        self.model_name = "B2_xgboost"

    def _build_estimator(self) -> Any:
        try:
            from xgboost import XGBRegressor
        except (ImportError, OSError) as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "xgboost is unavailable and allow_fallback=False"
                ) from exc
            self.used_fallback = True
            self.model_name = "B2_hist_gradient_boosting_fallback"
            max_iter = min(500, max(50, self.params["n_estimators"]))
            estimator = HistGradientBoostingRegressor(
                learning_rate=self.params["learning_rate"],
                max_iter=max_iter,
                max_depth=self.params["max_depth"],
                l2_regularization=self.params["reg_lambda"],
                random_state=self.params["random_state"],
            )
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("regressor", estimator),
                ]
            )
        self.used_fallback = False
        self.model_name = "B2_xgboost"
        params = {
            **self.params,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "verbosity": 0,
        }
        return XGBRegressor(**params)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> XGBoostLifetimeModel:
        target = validate_target(y)
        frame, columns = numeric_feature_frame(X)
        if len(frame) != target.size:
            raise ValueError("X and y have different row counts")
        self.estimator_ = self._build_estimator()
        self.estimator_.fit(frame, target)
        self.feature_names_in_ = columns
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ModelNotFittedError("XGBoostLifetimeModel must be fitted before predict")
        frame, _ = numeric_feature_frame(X, expected_columns=self.feature_names_in_)
        prediction = np.asarray(self.estimator_.predict(frame), dtype=float)
        return np.asarray(np.maximum(prediction, 1.0), dtype=float)
