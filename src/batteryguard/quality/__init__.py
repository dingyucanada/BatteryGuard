"""Data-quality and leakage gates."""

from batteryguard.quality.checks import DataQualityError, assert_data_quality, run_quality_checks
from batteryguard.quality.leakage import (
    DataLeakageError,
    assert_cell_level_split,
    assert_feature_frame_safe,
    assert_no_future_cycles,
    assert_no_leakage,
    assert_protocol_holdout,
    assert_training_scope,
)
from batteryguard.quality.report import build_quality_report

__all__ = [
    "DataLeakageError",
    "DataQualityError",
    "assert_cell_level_split",
    "assert_data_quality",
    "assert_feature_frame_safe",
    "assert_no_future_cycles",
    "assert_no_leakage",
    "assert_protocol_holdout",
    "assert_training_scope",
    "build_quality_report",
    "run_quality_checks",
]
