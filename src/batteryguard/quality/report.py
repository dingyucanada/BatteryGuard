"""Composition helper for the quality and leakage gates."""

from __future__ import annotations

import pandas as pd

from batteryguard.quality.checks import DataQualityError, run_quality_checks
from batteryguard.quality.leakage import DataLeakageError, assert_no_leakage
from batteryguard.schemas.data import DataQualityIssue, DataQualityReport


def build_quality_report(
    dataset_id: str,
    cells: pd.DataFrame,
    cycles: pd.DataFrame,
    timeseries: pd.DataFrame | None,
    splits: pd.DataFrame,
    *,
    features: pd.DataFrame | None = None,
    feature_source_cycles: pd.DataFrame | None = None,
    early_cycles: int | None = None,
    require_protocol_holdout: bool = False,
    hard_fail: bool = True,
) -> DataQualityReport:
    report = run_quality_checks(
        dataset_id,
        cells,
        cycles,
        timeseries,
        required_early_cycles=early_cycles,
    )
    try:
        assert_no_leakage(
            cells,
            splits,
            features=features,
            feature_source_cycles=feature_source_cycles,
            early_cycles=early_cycles,
            require_protocol_holdout=require_protocol_holdout,
        )
    except DataLeakageError as exc:
        report.issues.append(
            DataQualityIssue(code="LEAKAGE_GATE_FAILED", severity="ERROR", message=str(exc))
        )
        report.leakage_checks_passed = False
    else:
        report.leakage_checks_passed = True

    has_errors = any(issue.severity.upper() == "ERROR" for issue in report.issues)
    if has_errors:
        report.quality_score = min(report.quality_score, 0.5)
        if hard_fail:
            raise DataQualityError(report)
    return report
