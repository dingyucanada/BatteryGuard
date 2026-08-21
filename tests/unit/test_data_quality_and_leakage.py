from __future__ import annotations

import pandas as pd
import pytest

from batteryguard.ingestion.splits import build_cell_level_split, build_protocol_holdout_split
from batteryguard.quality import (
    DataLeakageError,
    DataQualityError,
    assert_cell_level_split,
    assert_data_quality,
    assert_feature_frame_safe,
    assert_no_future_cycles,
    assert_no_leakage,
    assert_protocol_holdout,
    assert_training_scope,
    build_quality_report,
)
from batteryguard.quality.leakage import assert_split_isolation


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.DataFrame(
        [
            {
                "cell_id": f"C{i}",
                "chemistry": "LFP_graphite",
                "nominal_capacity_ah": 1.1,
                "batch_id": "B",
                "protocol_id": f"P{i % 2}",
                "cycle_life": 500 + i,
                "eol_threshold": 0.8,
                "censored": False,
            }
            for i in range(6)
        ]
    )
    cycles = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "cycle_index": cycle,
                "charge_capacity_ah": 1.1,
                "discharge_capacity_ah": 1.08,
                "coulombic_efficiency": 0.99,
                "charge_time_s": 3600,
                "discharge_time_s": 3500,
                "dcir_ohm": None,
                "avg_temp_c": None,
                "max_temp_c": None,
                "charge_energy_wh": None,
                "discharge_energy_wh": None,
            }
            for cell_id in cells["cell_id"]
            for cycle in range(1, 4)
        ]
    )
    return cells, cycles


def test_quality_gate_passes_canonical_data_and_fails_duplicate() -> None:
    cells, cycles = _tables()
    report = assert_data_quality("fixture", cells, cycles)
    assert report.quality_score == 1.0
    bad = pd.concat([cycles, cycles.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataQualityError) as error:
        assert_data_quality("fixture", cells, bad)
    assert any(issue.code == "DUPLICATE_CYCLE" for issue in error.value.report.issues)


def test_leakage_guards_reject_labels_future_cycles_and_calibration_training() -> None:
    cells, cycles = _tables()
    with pytest.raises(DataLeakageError, match="leaky feature"):
        assert_feature_frame_safe(pd.DataFrame({"cell_id": ["C0"], "cycle_life": [500]}))
    with pytest.raises(DataLeakageError, match="after early_cycles"):
        assert_no_future_cycles(cycles, early_cycles=2)
    splits = build_cell_level_split(cells, seed=5)
    calibration_id = str(splits.loc[splits["split"] == "calibration", "cell_id"].iloc[0])
    with pytest.raises(DataLeakageError, match="only train split"):
        assert_training_scope([calibration_id], splits)


def test_split_duplicate_is_a_hard_leakage_error() -> None:
    cells, _ = _tables()
    splits = build_cell_level_split(cells)
    duplicate = splits.iloc[[0]].copy()
    duplicate["split"] = "test"
    corrupted = pd.concat([splits, duplicate], ignore_index=True)
    with pytest.raises(DataLeakageError, match="cell-level split violation"):
        assert_cell_level_split(corrupted)


def test_protocol_holdout_and_composed_leakage_paths() -> None:
    cells, cycles = _tables()
    splits = build_protocol_holdout_split(cells, holdout_protocols=["P1"])
    assert_protocol_holdout(cells, splits)
    assert_no_leakage(
        cells,
        splits,
        features=pd.DataFrame({"cell_id": cells["cell_id"], "safe": 1.0}),
        feature_source_cycles=cycles.loc[cycles["cycle_index"] <= 2],
        early_cycles=2,
        require_protocol_holdout=True,
        training_cell_ids=splits.loc[splits["split"] == "train", "cell_id"],
    )

    overlapping = build_cell_level_split(cells)
    with pytest.raises(DataLeakageError, match="test protocols overlap"):
        assert_protocol_holdout(cells, overlapping)
    with pytest.raises(DataLeakageError, match="missing protocol holdout"):
        assert_protocol_holdout(cells.drop(columns="protocol_id"), splits)
    with pytest.raises(DataLeakageError, match="split coverage differs"):
        assert_no_leakage(cells, splits.iloc[:-1])
    with pytest.raises(ValueError, match="early_cycles"):
        assert_no_leakage(cells, splits, feature_source_cycles=cycles)


def test_additional_leakage_validation_failures() -> None:
    cells, _ = _tables()
    splits = build_cell_level_split(cells)
    with pytest.raises(DataLeakageError, match="missing columns"):
        assert_cell_level_split(splits.drop(columns="split"))
    with pytest.raises(DataLeakageError, match="empty"):
        assert_cell_level_split(splits.iloc[0:0])
    with pytest.raises(DataLeakageError, match="unknown split names"):
        assert_cell_level_split(splits.assign(split="invalid"))
    with pytest.raises(ValueError, match="positive"):
        assert_no_future_cycles(pd.DataFrame({"cycle_index": [1]}), 0)
    with pytest.raises(DataLeakageError, match="missing cycle_index"):
        assert_no_future_cycles(pd.DataFrame({"cell_id": ["C0"]}), 2)
    with pytest.raises(DataLeakageError, match="non-numeric"):
        assert_no_future_cycles(pd.DataFrame({"cycle_index": ["bad"]}), 2)
    with pytest.raises(TypeError, match="DataFrame"):
        assert_feature_frame_safe([])  # type: ignore[arg-type]
    with pytest.raises(DataLeakageError, match="one row per cell"):
        assert_feature_frame_safe(pd.DataFrame({"cell_id": ["C0", "C0"], "safe": [1, 2]}))
    with pytest.raises(DataLeakageError, match="leaky feature"):
        assert_feature_frame_safe(pd.DataFrame({"cell_id": ["C0"], "cycle_31_mean": [1]}), early_cycles=30)
    with pytest.raises(DataLeakageError, match="absent from manifest"):
        assert_training_scope(["UNKNOWN"], splits)
    with pytest.raises(DataLeakageError, match="overlap"):
        assert_split_isolation({"train": ["C0"], "test": ["C0"]})
    assert_split_isolation({"train": ["C0"], "test": ["C1"]})


def test_quality_report_can_return_leakage_failure_without_raising() -> None:
    cells, cycles = _tables()
    splits = build_cell_level_split(cells)
    report = build_quality_report(
        "fixture",
        cells,
        cycles,
        None,
        splits,
        features=pd.DataFrame({"cell_id": cells["cell_id"], "cycle_life": cells["cycle_life"]}),
        hard_fail=False,
    )
    assert not report.leakage_checks_passed
    assert report.quality_score <= 0.5
    with pytest.raises(DataQualityError):
        build_quality_report(
            "fixture",
            cells,
            cycles,
            None,
            splits,
            features=pd.DataFrame({"cycle_life": [500]}),
            hard_fail=True,
        )
