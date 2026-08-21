from __future__ import annotations

import numpy as np
import pytest

from batteryguard.uncertainty import (
    ConformalNotCalibratedError,
    SplitConformalRegressor,
    empirical_coverage,
    finite_sample_quantile,
)


def test_finite_sample_quantile_uses_higher_order_statistic() -> None:
    scores = np.arange(1.0, 11.0)
    assert finite_sample_quantile(scores, alpha=0.2) == 9.0
    assert finite_sample_quantile(scores, alpha=0.1) == 10.0


def test_split_conformal_calibrates_and_reports_coverage() -> None:
    truth = np.asarray([100, 110, 120, 130, 140], dtype=float)
    predicted = np.asarray([101, 108, 123, 126, 145], dtype=float)
    conformal = SplitConformalRegressor(alpha=0.2).calibrate(truth, predicted)
    low, high = conformal.interval(predicted)
    coverage, width = empirical_coverage(truth, low, high)
    assert conformal.quantile_ == 5.0
    assert coverage == 1.0
    assert width == 10.0


def test_interval_before_calibration_fails() -> None:
    with pytest.raises(ConformalNotCalibratedError):
        SplitConformalRegressor().interval([100.0])


def test_conformal_validation_predict_interval_and_bad_coverage() -> None:
    for alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha"):
            SplitConformalRegressor(alpha=alpha)
        with pytest.raises(ValueError, match="alpha"):
            finite_sample_quantile([1.0], alpha)
    with pytest.raises(ValueError, match="empty"):
        finite_sample_quantile([], 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        finite_sample_quantile([1.0, np.nan], 0.1)
    with pytest.raises(ValueError, match="negative"):
        finite_sample_quantile([-1.0], 0.1)
    conformal = SplitConformalRegressor(alpha=0.2, clip_lower=None)
    with pytest.raises(ValueError, match="different lengths"):
        conformal.calibrate([1.0], [1.0, 2.0])
    conformal.calibrate([10.0, 20.0], [9.0, 22.0])

    class Predictor:
        def predict(self, X: object) -> np.ndarray:
            return np.asarray([15.0, 25.0])

    point, low, high = conformal.predict_interval(Predictor(), object())
    assert np.array_equal(point, [15.0, 25.0])
    assert np.all(low < point)
    assert np.all(high > point)
    with pytest.raises(ValueError, match="different lengths"):
        empirical_coverage([1.0], [0.0, 1.0], [2.0])
    with pytest.raises(ValueError, match="cannot exceed"):
        empirical_coverage([1.0], [2.0], [0.0])
