"""B0 median and B1 regularized-linear lifetime baselines."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from batteryguard.quality.leakage import assert_feature_frame_safe


class ModelNotFittedError(RuntimeError):
    """Raised when prediction is attempted before fit."""


def validate_target(y: Any) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("target cannot be empty")
    if np.any(~np.isfinite(values)):
        raise ValueError("target contains missing/non-finite values; censored cells need explicit handling")
    if np.any(values <= 0):
        raise ValueError("cycle-life target must be positive")
    return values


def numeric_feature_frame(
    X: pd.DataFrame | np.ndarray,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        assert_feature_frame_safe(X)
        frame = X.drop(columns=["cell_id"], errors="ignore").copy()
        if expected_columns is None:
            columns = tuple(str(column) for column in frame.columns)
            if not columns:
                raise ValueError("feature frame has no model columns")
        else:
            columns = expected_columns
            missing = [column for column in columns if column not in frame.columns]
            extra = [str(column) for column in frame.columns if str(column) not in columns]
            if missing or extra:
                raise ValueError(f"feature columns differ from fit (missing={missing}, extra={extra})")
            frame = frame.loc[:, list(columns)]
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.astype(float), columns
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    columns = expected_columns or tuple(f"feature_{index}" for index in range(array.shape[1]))
    if len(columns) != array.shape[1]:
        raise ValueError("array feature count differs from fit")
    return pd.DataFrame(array, columns=list(columns)), columns


class MedianLifetimeModel:
    """B0 constant predictor: median observed train lifetime."""

    model_name = "B0_train_median"

    def __init__(self) -> None:
        self.median_: float | None = None
        self.feature_names_in_: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> MedianLifetimeModel:
        target = validate_target(y)
        frame, columns = numeric_feature_frame(X)
        if len(frame) != target.size:
            raise ValueError("X and y have different row counts")
        self.median_ = float(np.median(target))
        self.feature_names_in_ = columns
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise ModelNotFittedError("MedianLifetimeModel must be fitted before predict")
        frame, _ = numeric_feature_frame(X, expected_columns=self.feature_names_in_)
        return np.full(len(frame), self.median_, dtype=float)


class LinearLifetimeModel:
    """B1 median-imputed, standardized Ridge or Elastic Net model."""

    def __init__(
        self,
        *,
        kind: Literal["ridge", "elasticnet"] = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.15,
        max_iter: int = 20_000,
        random_state: int = 42,
    ) -> None:
        if kind not in {"ridge", "elasticnet"}:
            raise ValueError("kind must be 'ridge' or 'elasticnet'")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be in [0, 1]")
        self.kind = kind
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        estimator: Ridge | ElasticNet
        if kind == "ridge":
            estimator = Ridge(alpha=self.alpha)
        else:
            estimator = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=self.random_state,
                selection="cyclic",
            )
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("regressor", estimator),
            ]
        )
        self.feature_names_in_: tuple[str, ...] = ()
        self.is_fitted_ = False
        self.model_name = f"B1_{kind}"

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> LinearLifetimeModel:
        target = validate_target(y)
        frame, columns = numeric_feature_frame(X)
        if len(frame) != target.size:
            raise ValueError("X and y have different row counts")
        self.pipeline.fit(frame, target)
        self.feature_names_in_ = columns
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ModelNotFittedError("LinearLifetimeModel must be fitted before predict")
        frame, _ = numeric_feature_frame(X, expected_columns=self.feature_names_in_)
        prediction = np.asarray(self.pipeline.predict(frame), dtype=float)
        return np.maximum(prediction, 1.0)


class RidgeLifetimeModel(LinearLifetimeModel):
    def __init__(self, *, alpha: float = 1.0, **kwargs: Any) -> None:
        super().__init__(kind="ridge", alpha=alpha, **kwargs)


class ElasticNetLifetimeModel(LinearLifetimeModel):
    def __init__(
        self, *, alpha: float = 0.01, l1_ratio: float = 0.15, **kwargs: Any
    ) -> None:
        super().__init__(kind="elasticnet", alpha=alpha, l1_ratio=l1_ratio, **kwargs)
