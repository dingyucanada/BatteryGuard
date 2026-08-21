"""Distribution-free split conformal intervals for a frozen point model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np


class PointPredictor(Protocol):
    def predict(self, X: Any) -> np.ndarray: ...


class ConformalNotCalibratedError(RuntimeError):
    """Raised when an interval is requested before held-out calibration."""


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def finite_sample_quantile(scores: Iterable[float], alpha: float) -> float:
    """Return score at rank ceil((n+1)*(1-alpha)), capped at n.

    This is the conformal ``higher`` order statistic, not NumPy's interpolated
    sample quantile, and thus preserves the finite-sample coverage guarantee.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = _finite_vector(scores, "scores")
    if np.any(values < 0):
        raise ValueError("nonconformity scores cannot be negative")
    rank = min(values.size, int(np.ceil((values.size + 1) * (1.0 - alpha))))
    return float(np.partition(values, rank - 1)[rank - 1])


class SplitConformalRegressor:
    """Calibrate absolute residuals from a strictly held-out calibration split."""

    def __init__(self, alpha: float = 0.10, *, clip_lower: float | None = 0.0) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.clip_lower = float(clip_lower) if clip_lower is not None else None
        self.quantile_: float | None = None
        self.calibration_size_: int = 0
        self.calibration_scores_: np.ndarray | None = None

    @property
    def coverage_target(self) -> float:
        return 1.0 - self.alpha

    def calibrate(
        self,
        y_true: Iterable[float],
        y_pred: Iterable[float],
    ) -> SplitConformalRegressor:
        truth = _finite_vector(y_true, "y_true")
        prediction = _finite_vector(y_pred, "y_pred")
        if truth.size != prediction.size:
            raise ValueError("y_true and y_pred have different lengths")
        scores = np.abs(truth - prediction)
        self.quantile_ = finite_sample_quantile(scores, self.alpha)
        self.calibration_size_ = int(scores.size)
        self.calibration_scores_ = scores.copy()
        return self

    fit = calibrate

    def interval(self, y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile_ is None:
            raise ConformalNotCalibratedError("calibrate on held-out residuals before interval")
        prediction = _finite_vector(y_pred, "y_pred")
        low = prediction - self.quantile_
        if self.clip_lower is not None:
            low = np.maximum(low, self.clip_lower)
        high = prediction + self.quantile_
        return low, high

    def predict_interval(
        self, model: PointPredictor, X: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = np.asarray(model.predict(X), dtype=float).reshape(-1)
        low, high = self.interval(prediction)
        return prediction, low, high


def empirical_coverage(
    y_true: Iterable[float],
    interval_low: Iterable[float],
    interval_high: Iterable[float],
) -> tuple[float, float]:
    truth = _finite_vector(y_true, "y_true")
    low = _finite_vector(interval_low, "interval_low")
    high = _finite_vector(interval_high, "interval_high")
    if truth.size != low.size or truth.size != high.size:
        raise ValueError("coverage arrays have different lengths")
    if np.any(low > high):
        raise ValueError("interval_low cannot exceed interval_high")
    covered = (truth >= low) & (truth <= high)
    return float(np.mean(covered)), float(np.mean(high - low))
