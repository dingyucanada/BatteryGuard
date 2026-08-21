"""Shrinkage-covariance Mahalanobis OOD detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from batteryguard.prediction.baselines import numeric_feature_frame


class MahalanobisNotFittedError(RuntimeError):
    """Raised when OOD scores are requested before fitting the train split."""


class MahalanobisOOD:
    """Fit only on training features and map squared distance to [0, 1]."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.covariance = LedoitWolf(assume_centered=False)
        self.feature_names_in_: tuple[str, ...] = ()
        self.training_distances_: np.ndarray | None = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> MahalanobisOOD:
        frame, columns = numeric_feature_frame(X)
        if len(frame) < 3:
            raise ValueError("Mahalanobis OOD requires at least three training cells")
        imputed = self.imputer.fit_transform(frame)
        transformed = self.scaler.fit_transform(imputed)
        self.covariance.fit(transformed)
        squared = np.asarray(self.covariance.mahalanobis(transformed), dtype=float)
        self.training_distances_ = np.sqrt(np.maximum(squared, 0.0))
        self.feature_names_in_ = columns
        self.is_fitted_ = True
        return self

    def _transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise MahalanobisNotFittedError("fit MahalanobisOOD before scoring")
        frame, _ = numeric_feature_frame(X, expected_columns=self.feature_names_in_)
        return np.asarray(self.scaler.transform(self.imputer.transform(frame)), dtype=float)

    def distance(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        transformed = self._transform(X)
        squared = np.asarray(self.covariance.mahalanobis(transformed), dtype=float)
        return np.sqrt(np.maximum(squared, 0.0))

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return larger-is-more-OOD probability-like scores in [0, 1]."""

        distance = self.distance(X)
        if self.training_distances_ is None:
            raise MahalanobisNotFittedError("fit MahalanobisOOD before scoring")
        empirical = np.searchsorted(
            np.sort(self.training_distances_), distance, side="right"
        ) / (self.training_distances_.size + 1.0)
        degrees = max(1, len(self.feature_names_in_))
        parametric = chi2.cdf(np.square(distance), df=degrees)
        return np.asarray(np.clip(np.maximum(empirical, parametric), 0.0, 1.0), dtype=float)

    def is_ood(self, X: pd.DataFrame | np.ndarray, *, threshold: float = 0.98) -> np.ndarray:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        return self.score_samples(X) >= threshold

    predict_proba = score_samples
