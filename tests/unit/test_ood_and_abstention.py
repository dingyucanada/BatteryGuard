from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batteryguard.ood import (
    AbstentionPolicy,
    MahalanobisNotFittedError,
    MahalanobisOOD,
    should_abstain,
)


def test_mahalanobis_scores_far_query_above_training_cluster() -> None:
    rng = np.random.default_rng(7)
    train = pd.DataFrame(rng.normal(0, 0.2, size=(30, 3)), columns=["a", "b", "c"])
    detector = MahalanobisOOD().fit(train)
    near = detector.score_samples(pd.DataFrame([[0.0, 0.0, 0.0]], columns=train.columns))[0]
    far = detector.score_samples(pd.DataFrame([[8.0, 8.0, 8.0]], columns=train.columns))[0]
    assert 0 <= near < far <= 1
    assert detector.is_ood(pd.DataFrame([[8.0, 8.0, 8.0]], columns=train.columns))[0]


def test_abstention_accumulates_all_reasons() -> None:
    decision = AbstentionPolicy(required_early_cycles=30).evaluate(
        quality_score=0.5,
        ood_score=0.999,
        point_estimate=100,
        interval_low=60,
        interval_high=150,
        observed_cycles=10,
        protocol_available=False,
        model_version="m1",
        evidence_model_version="m2",
    )
    assert decision.abstain
    assert {
        "LOW_DATA_QUALITY",
        "OUT_OF_DISTRIBUTION",
        "INTERVAL_TOO_WIDE",
        "INSUFFICIENT_EARLY_CYCLES",
        "MISSING_PROTOCOL_METADATA",
        "MODEL_EVIDENCE_VERSION_MISMATCH",
    }.issubset(decision.reasons)


def test_ood_and_abstention_validation_paths() -> None:
    detector = MahalanobisOOD()
    with pytest.raises(ValueError, match="at least three"):
        detector.fit(np.ones((2, 2)))
    with pytest.raises(MahalanobisNotFittedError):
        detector.distance(np.ones((1, 2)))
    train = pd.DataFrame(np.eye(3), columns=["a", "b", "c"])
    detector.fit(train)
    assert detector.predict_proba(train.iloc[[0]]).shape == (1,)
    with pytest.raises(ValueError, match="threshold"):
        detector.is_ood(train.iloc[[0]], threshold=0)
    for kwargs in (
        {"quality_threshold": -1},
        {"ood_threshold": 0},
        {"interval_width_ratio_max": 0},
        {"required_early_cycles": 0},
    ):
        with pytest.raises(ValueError):
            AbstentionPolicy(**kwargs)
    missing = AbstentionPolicy(required_early_cycles=1).evaluate(
        quality_score=None,
        ood_score=None,
        point_estimate=None,
        interval_low=None,
        interval_high=None,
        observed_cycles=1,
        protocol_available=True,
    )
    assert {
        "MISSING_DATA_QUALITY",
        "MISSING_OOD_SCORE",
        "MISSING_OR_INVALID_INTERVAL",
    }.issubset(missing.reasons)
    malformed = should_abstain(
        quality_score=1.0,
        ood_score=0.0,
        point_estimate=100.0,
        interval_low=110.0,
        interval_high=120.0,
        observed_cycles=30,
        protocol_available=True,
    )
    assert malformed.reasons == ("MALFORMED_INTERVAL",)
