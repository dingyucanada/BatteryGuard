from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batteryguard.prediction import (
    ElasticNetLifetimeModel,
    MedianLifetimeModel,
    ModelLadder,
    ModelNotFittedError,
    RidgeLifetimeModel,
    XGBoostLifetimeModel,
    build_model,
    evaluate_regression,
    grouped_evaluate,
)
from batteryguard.prediction.baselines import LinearLifetimeModel, numeric_feature_frame
from batteryguard.quality import DataLeakageError


def _regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    x = np.linspace(-2.0, 2.0, 40)
    frame = pd.DataFrame(
        {
            "cell_id": [f"C{i}" for i in range(40)],
            "capacity_slope": x,
            "temperature": np.square(x),
            "optional": np.where(np.arange(40) % 5 == 0, np.nan, x / 2),
        }
    )
    target = 800 + 120 * x + 5 * np.square(x)
    return frame, target


def test_b0_b1_and_b2_have_common_fit_predict_contract() -> None:
    X, y = _regression_data()
    median = MedianLifetimeModel().fit(X, y)
    assert np.allclose(median.predict(X.iloc[:3]), np.median(y))
    ridge = RidgeLifetimeModel(alpha=0.01).fit(X, y)
    ridge_metrics = evaluate_regression(y, ridge.predict(X)).overall
    assert ridge_metrics.mae < 5.0
    boosted = XGBoostLifetimeModel(n_estimators=40).fit(X, y)
    prediction = boosted.predict(X.iloc[:4])
    assert prediction.shape == (4,)
    assert boosted.model_name in {"B2_xgboost", "B2_hist_gradient_boosting_fallback"}


def test_model_ladder_preserves_visible_levels() -> None:
    X, y = _regression_data()
    ladder = ModelLadder(model_kwargs={"B2": {"n_estimators": 30}}).fit(X, y)
    output = ladder.predict_all(X.iloc[:2])
    assert list(output.columns) == ["B0", "B1", "B2"]


def test_model_rejects_leaky_label_column() -> None:
    X, y = _regression_data()
    X["cycle_life"] = y
    with pytest.raises(DataLeakageError):
        RidgeLifetimeModel().fit(X, y)


def test_baseline_input_validation_and_elastic_net_paths() -> None:
    X, y = _regression_data()
    elastic = ElasticNetLifetimeModel(alpha=0.001, l1_ratio=0.3).fit(X, y)
    assert np.isfinite(elastic.predict(X.iloc[:2])).all()
    array_frame, names = numeric_feature_frame(np.asarray([[1.0, 2.0]]))
    assert array_frame.shape == (1, 2)
    assert names == ("feature_0", "feature_1")
    with pytest.raises(ValueError, match="2D"):
        numeric_feature_frame(np.empty((1, 0)))
    with pytest.raises(ValueError, match="no model columns"):
        numeric_feature_frame(pd.DataFrame({"cell_id": ["C"]}))
    with pytest.raises(ValueError, match="differ from fit"):
        numeric_feature_frame(pd.DataFrame({"other": [1.0]}), expected_columns=("expected",))
    with pytest.raises(ValueError, match="array feature count"):
        numeric_feature_frame(np.asarray([[1.0, 2.0]]), expected_columns=("one",))
    with pytest.raises(ValueError, match="kind"):
        LinearLifetimeModel(kind="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        RidgeLifetimeModel(alpha=0)
    with pytest.raises(ValueError, match="l1_ratio"):
        LinearLifetimeModel(kind="elasticnet", l1_ratio=2)
    with pytest.raises(ValueError, match="empty"):
        MedianLifetimeModel().fit(np.ones((0, 1)), [])
    with pytest.raises(ValueError, match="non-finite"):
        MedianLifetimeModel().fit(np.ones((2, 1)), [100.0, np.nan])
    with pytest.raises(ValueError, match="positive"):
        MedianLifetimeModel().fit(np.ones((2, 1)), [100.0, 0.0])
    with pytest.raises(ModelNotFittedError):
        MedianLifetimeModel().predict(np.ones((1, 1)))
    with pytest.raises(ModelNotFittedError):
        RidgeLifetimeModel().predict(np.ones((1, 1)))
    with pytest.raises(ValueError, match="different row counts"):
        RidgeLifetimeModel().fit(X.iloc[:2], y[:3])


def test_model_registry_and_ladder_error_paths() -> None:
    assert isinstance(build_model("B0"), MedianLifetimeModel)
    assert isinstance(build_model("elastic-net"), ElasticNetLifetimeModel)
    assert isinstance(build_model("B2"), XGBoostLifetimeModel)
    with pytest.raises(ValueError, match="no hyperparameters"):
        build_model("median", alpha=1)
    with pytest.raises(ValueError, match="unknown model"):
        build_model("neural-net")
    ladder = ModelLadder()
    with pytest.raises(RuntimeError, match="fitted"):
        ladder.predict_all(np.ones((1, 2)))
    with pytest.raises(RuntimeError, match="fitted"):
        ladder.best_available()
    X, y = _regression_data()
    assert ladder.fit(X, y).best_available() is ladder.models["B2"]


def test_evaluation_validation_and_ood_subsets() -> None:
    truth = [100.0, 200.0, 300.0, 400.0]
    prediction = [110.0, 190.0, 330.0, 390.0]
    report = evaluate_regression(
        truth,
        prediction,
        groups=["A", "A", "B", "B"],
        lifetime_quantiles=2,
        ood_mask=[False, False, True, True],
    )
    assert report.in_domain is not None
    assert report.ood is not None
    assert report.ood_mae_gap is not None
    with pytest.raises(ValueError, match="equal length"):
        evaluate_regression([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        evaluate_regression([1.0], [np.nan])
    with pytest.raises(ValueError, match="positive"):
        evaluate_regression(truth, prediction, lifetime_quantiles=0)
    with pytest.raises(ValueError, match="groups length"):
        evaluate_regression(truth, prediction, groups=["A"])
    with pytest.raises(ValueError, match="ood_mask length"):
        evaluate_regression(truth, prediction, ood_mask=[True])
    with pytest.raises(ValueError, match="equal length"):
        grouped_evaluate(lambda: RidgeLifetimeModel(), np.ones((2, 1)), [1.0], ["A"])
    with pytest.raises(ValueError, match="at least two"):
        grouped_evaluate(
            lambda: RidgeLifetimeModel(), np.ones((2, 1)), [1.0, 2.0], ["A", "A"]
        )
