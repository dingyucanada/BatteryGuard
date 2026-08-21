"""Numerically stable trend primitives used by the early-cycle pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _finite_xy(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(list(x), dtype=float).reshape(-1)
    y_array = np.asarray(list(y), dtype=float).reshape(-1)
    if x_array.size != y_array.size:
        raise ValueError("x and y must have equal length")
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    return x_array[mask], y_array[mask]


def linear_trend(x: Iterable[float], y: Iterable[float]) -> float:
    """Least-squares slope after centering x to limit condition-number issues."""

    x_array, y_array = _finite_xy(x, y)
    if x_array.size < 2 or np.ptp(x_array) <= 0:
        return float("nan")
    centered = x_array - np.mean(x_array)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(centered, y_array - np.mean(y_array)) / denominator)


def quadratic_curvature(x: Iterable[float], y: Iterable[float]) -> float:
    """Return the quadratic coefficient in y = a*x^2 + b*x + c."""

    x_array, y_array = _finite_xy(x, y)
    if x_array.size < 3 or np.unique(x_array).size < 3:
        return float("nan")
    scale = float(np.std(x_array))
    if scale <= 0:
        return float("nan")
    centered = (x_array - np.mean(x_array)) / scale
    coefficient = np.polyfit(centered, y_array, deg=2)[0]
    return float(coefficient / (scale * scale))


def trend_statistics(x: Iterable[float], y: Iterable[float], prefix: str) -> dict[str, float]:
    x_array, y_array = _finite_xy(x, y)
    if y_array.size == 0:
        return {
            f"{prefix}_slope_per_cycle": float("nan"),
            f"{prefix}_curvature_per_cycle2": float("nan"),
            f"{prefix}_first": float("nan"),
            f"{prefix}_last": float("nan"),
            f"{prefix}_change": float("nan"),
        }
    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = y_array[order]
    return {
        f"{prefix}_slope_per_cycle": linear_trend(x_array, y_array),
        f"{prefix}_curvature_per_cycle2": quadratic_curvature(x_array, y_array),
        f"{prefix}_first": float(y_array[0]),
        f"{prefix}_last": float(y_array[-1]),
        f"{prefix}_change": float(y_array[-1] - y_array[0]),
    }
