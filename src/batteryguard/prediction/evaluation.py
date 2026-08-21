"""Locked regression metrics and grouped, cell-disjoint evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold


class Regressor(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    count: int
    mae: float
    mape: float
    rmse: float
    spearman: float
    bias: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    overall: RegressionMetrics
    by_group: Mapping[str, RegressionMetrics]
    by_lifetime_quantile: Mapping[str, RegressionMetrics]
    in_domain: RegressionMetrics | None = None
    ood: RegressionMetrics | None = None
    ood_mae_gap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroupedEvaluationResult:
    predictions: np.ndarray
    fold_index: np.ndarray
    report: EvaluationReport


def _arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(list(y_true), dtype=float).reshape(-1)
    prediction = np.asarray(list(y_pred), dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != prediction.size:
        raise ValueError("y_true and y_pred must be non-empty with equal length")
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(prediction)):
        raise ValueError("metrics require finite y_true and y_pred")
    return truth, prediction


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> RegressionMetrics:
    truth, prediction = _arrays(y_true, y_pred)
    error = prediction - truth
    denominator_mask = np.abs(truth) > np.finfo(float).eps
    mape = (
        float(np.mean(np.abs(error[denominator_mask] / truth[denominator_mask])))
        if np.any(denominator_mask)
        else float("nan")
    )
    if truth.size < 2 or np.unique(truth).size < 2 or np.unique(prediction).size < 2:
        rank = 0.0
    else:
        statistic = spearmanr(truth, prediction, nan_policy="raise").statistic
        rank = float(statistic) if np.isfinite(statistic) else 0.0
    return RegressionMetrics(
        count=int(truth.size),
        mae=float(np.mean(np.abs(error))),
        mape=mape,
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        spearman=rank,
        bias=float(np.mean(error)),
    )


def _subset_metrics(
    truth: np.ndarray, prediction: np.ndarray, labels: np.ndarray
) -> dict[str, RegressionMetrics]:
    result: dict[str, RegressionMetrics] = {}
    for label in sorted({str(value) for value in labels}):
        mask = np.asarray([str(value) == label for value in labels], dtype=bool)
        result[label] = regression_metrics(truth[mask], prediction[mask])
    return result


def evaluate_regression(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    groups: Iterable[object] | None = None,
    lifetime_quantiles: int = 4,
    ood_mask: Iterable[bool] | None = None,
) -> EvaluationReport:
    truth, prediction = _arrays(y_true, y_pred)
    if lifetime_quantiles < 1:
        raise ValueError("lifetime_quantiles must be positive")
    by_group: dict[str, RegressionMetrics] = {}
    if groups is not None:
        group_array = np.asarray(list(groups), dtype=object).reshape(-1)
        if group_array.size != truth.size:
            raise ValueError("groups length differs from y_true")
        by_group = _subset_metrics(truth, prediction, group_array)

    try:
        quantile_labels = pd.qcut(
            truth,
            q=min(lifetime_quantiles, truth.size),
            labels=False,
            duplicates="drop",
        )
        quantile_array = np.asarray(quantile_labels, dtype=int)
    except ValueError:
        quantile_array = np.zeros(truth.size, dtype=int)
    by_quantile = _subset_metrics(
        truth,
        prediction,
        np.asarray([f"Q{value + 1}" for value in quantile_array], dtype=object),
    )

    in_domain_metrics: RegressionMetrics | None = None
    ood_metrics: RegressionMetrics | None = None
    gap: float | None = None
    if ood_mask is not None:
        mask = np.asarray(list(ood_mask), dtype=bool).reshape(-1)
        if mask.size != truth.size:
            raise ValueError("ood_mask length differs from y_true")
        if np.any(~mask):
            in_domain_metrics = regression_metrics(truth[~mask], prediction[~mask])
        if np.any(mask):
            ood_metrics = regression_metrics(truth[mask], prediction[mask])
        if in_domain_metrics is not None and ood_metrics is not None:
            gap = float(ood_metrics.mae - in_domain_metrics.mae)
    return EvaluationReport(
        overall=regression_metrics(truth, prediction),
        by_group=by_group,
        by_lifetime_quantile=by_quantile,
        in_domain=in_domain_metrics,
        ood=ood_metrics,
        ood_mae_gap=gap,
    )


def _take_rows(X: pd.DataFrame | np.ndarray, indices: np.ndarray) -> pd.DataFrame | np.ndarray:
    return X.iloc[indices] if isinstance(X, pd.DataFrame) else np.asarray(X)[indices]


def grouped_evaluate(
    model_factory: Callable[[], Regressor],
    X: pd.DataFrame | np.ndarray,
    y: Iterable[float],
    groups: Iterable[object],
    *,
    n_splits: int = 5,
) -> GroupedEvaluationResult:
    """Out-of-fold prediction with GroupKFold and explicit overlap assertions."""

    target = np.asarray(list(y), dtype=float).reshape(-1)
    group_array = np.asarray(list(groups), dtype=object).reshape(-1)
    if len(X) != target.size or group_array.size != target.size:
        raise ValueError("X, y, and groups must have equal length")
    unique_groups = np.unique(group_array)
    folds = min(int(n_splits), unique_groups.size)
    if folds < 2:
        raise ValueError("grouped evaluation requires at least two distinct groups")
    predictions = np.full(target.size, np.nan, dtype=float)
    fold_index = np.full(target.size, -1, dtype=int)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_index, test_index) in enumerate(splitter.split(np.zeros(target.size), target, group_array)):
        train_groups = set(group_array[train_index])
        test_groups = set(group_array[test_index])
        if train_groups & test_groups:
            raise RuntimeError("GroupKFold produced overlapping groups")
        model = model_factory()
        model.fit(_take_rows(X, train_index), target[train_index])
        predictions[test_index] = np.asarray(model.predict(_take_rows(X, test_index)), dtype=float)
        fold_index[test_index] = fold
    if np.any(~np.isfinite(predictions)) or np.any(fold_index < 0):
        raise RuntimeError("grouped evaluation did not predict every row")
    return GroupedEvaluationResult(
        predictions=predictions,
        fold_index=fold_index,
        report=evaluate_regression(target, predictions, groups=group_array),
    )


grouped_cross_validate = grouped_evaluate
