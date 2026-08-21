"""Model construction and visible model-ladder orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from batteryguard.prediction.baselines import (
    ElasticNetLifetimeModel,
    MedianLifetimeModel,
    RidgeLifetimeModel,
)
from batteryguard.prediction.xgboost_model import XGBoostLifetimeModel


def build_model(name: str, **kwargs: Any) -> Any:
    normalized = name.lower().replace("_", "-").strip()
    if normalized in {"b0", "median", "train-median"}:
        if kwargs:
            raise ValueError("median baseline accepts no hyperparameters")
        return MedianLifetimeModel()
    if normalized in {"b1", "ridge", "b1-ridge"}:
        return RidgeLifetimeModel(**kwargs)
    if normalized in {"elasticnet", "elastic-net", "b1-elasticnet"}:
        return ElasticNetLifetimeModel(**kwargs)
    if normalized in {"b2", "xgboost", "xgb", "boosted-tree"}:
        return XGBoostLifetimeModel(**kwargs)
    raise ValueError(f"unknown model {name!r}; choose median, ridge, elasticnet, or xgboost")


class ModelLadder:
    """Fit and retain B0/B1/B2 on exactly the same training rows."""

    def __init__(self, *, model_kwargs: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        options = dict(model_kwargs or {})
        self.models: dict[str, Any] = {
            "B0": build_model("median", **dict(options.get("B0", {}))),
            "B1": build_model("ridge", **dict(options.get("B1", {}))),
            "B2": build_model("xgboost", **dict(options.get("B2", {}))),
        }
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> ModelLadder:
        for model in self.models.values():
            model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_all(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("ModelLadder must be fitted before predict_all")
        return pd.DataFrame(
            {
                level: np.asarray(model.predict(X), dtype=float)
                for level, model in self.models.items()
            }
        )

    def best_available(self) -> Any:
        """Return B2; selection by metrics remains an explicit caller decision."""

        if not self.is_fitted_:
            raise RuntimeError("ModelLadder must be fitted before best_available")
        return self.models["B2"]
