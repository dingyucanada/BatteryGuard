from __future__ import annotations

import pandas as pd
import pytest

from batteryguard.features import (
    EarlyCycleFeaturePipeline,
    compute_delta_q_features,
    delta_q_curve,
    derive_protocol_features,
    linear_trend,
    parse_c_rates,
    protocol_stage_features,
    quadratic_curvature,
)
from batteryguard.features.delta_q import DeltaQError
from batteryguard.features.pipeline import FeatureExtractionError


def _feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.DataFrame(
        {
            "cell_id": ["A", "B", "C"],
            "nominal_capacity_ah": [1.0, 1.0, 1.0],
        }
    )
    cycles: list[dict[str, object]] = []
    series: list[dict[str, object]] = []
    for cell_number, cell_id in enumerate(cells["cell_id"]):
        for cycle in range(1, 7):
            capacity = 1.0 - cycle * (0.002 + cell_number * 0.0005)
            cycles.append(
                {
                    "cell_id": cell_id,
                    "cycle_index": cycle,
                    "charge_capacity_ah": capacity / 0.998,
                    "discharge_capacity_ah": capacity,
                    "coulombic_efficiency": 0.998 - cell_number * 0.0002,
                    "charge_time_s": 3600 + cycle * 4,
                    "discharge_time_s": 3400,
                    "dcir_ohm": 0.02 + cycle * 0.0001,
                    "avg_temp_c": 25 + cell_number,
                    "max_temp_c": 28 + cell_number,
                    "charge_energy_wh": None,
                    "discharge_energy_wh": None,
                }
            )
            for sample in range(12):
                discharge = sample >= 3
                discharge_step = sample - 3
                series.append(
                    {
                        "cell_id": cell_id,
                        "cycle_index": cycle,
                        "sample_index": sample,
                        "time_s": sample * 300,
                        "current_a": -1.0 if discharge else 1.5,
                        "voltage_v": (3.5 - discharge_step * 0.08) if discharge else 3.0 + sample * 0.1,
                        "charge_capacity_ah": min(sample / 3, 1.0),
                        "discharge_capacity_ah": max(discharge_step, 0) / 8 * capacity,
                        "temperature_c": 25 + cell_number,
                        "step_type": "discharge" if discharge else "charge",
                    }
                )
    return cells, pd.DataFrame(cycles), pd.DataFrame(series)


def test_pipeline_uses_only_early_window_and_builds_trends_and_delta_q() -> None:
    cells, cycles, timeseries = _feature_tables()
    pipeline = EarlyCycleFeaturePipeline(
        early_cycles=5,
        reference_cycle=2,
        min_valid_cycles=3,
    )
    features = pipeline.transform(cells, cycles, timeseries)
    assert len(features) == 3
    assert "cycle_life" not in features
    assert "protocol_id" not in features
    assert features["discharge_capacity_ah_slope_per_cycle"].lt(0).all()
    assert features["delta_q_comparison_cycle"].eq(5).all()
    assert features["charge_c_rate_peak"].gt(1.0).all()
    assert pipeline.feature_version.startswith("early5-dq-")

    changed_future = cycles.copy()
    changed_future.loc[changed_future["cycle_index"] == 6, "discharge_capacity_ah"] = 99.0
    rerun = pipeline.transform(cells, changed_future, timeseries)
    pd.testing.assert_frame_equal(features, rerun)


def test_delta_q_curve_shift_has_expected_sign() -> None:
    _, _, timeseries = _feature_tables()
    features = compute_delta_q_features(
        timeseries,
        early_cycles=5,
        reference_cycle=2,
        comparison_cycle=5,
    )
    assert len(features) == 3
    assert features["delta_q_mean_ah"].lt(0).all()
    assert features["delta_q_l1_ah"].gt(0).all()


def test_delta_q_validates_curve_support_grid_and_strict_failures() -> None:
    reference = pd.DataFrame(
        {
            "voltage_v": [3.0, 3.1, 3.2, 3.3],
            "discharge_capacity_ah": [0.0, 0.3, 0.6, 0.9],
            "current_a": [-1.0] * 4,
        }
    )
    comparison = reference.assign(discharge_capacity_ah=[0.0, 0.29, 0.58, 0.87])
    grid, delta = delta_q_curve(
        reference,
        comparison,
        voltage_grid=[3.0, 3.15, 3.3],
    )
    assert len(grid) == len(delta) == 3
    with pytest.raises(ValueError, match="grid_points"):
        delta_q_curve(reference, comparison, grid_points=4)
    with pytest.raises(DeltaQError, match="increasing finite"):
        delta_q_curve(reference, comparison, voltage_grid=[3.0, 3.2, 3.1])
    with pytest.raises(DeltaQError, match="outside common"):
        delta_q_curve(reference, comparison, voltage_grid=[2.9, 3.1, 3.3])
    with pytest.raises(DeltaQError, match="no overlapping"):
        delta_q_curve(reference, reference.assign(voltage_v=[4.0, 4.1, 4.2, 4.3]))
    with pytest.raises(DeltaQError, match="at least three"):
        delta_q_curve(reference.iloc[:2], comparison)
    with pytest.raises(DeltaQError, match="distinct voltage"):
        delta_q_curve(reference.assign(voltage_v=3.0), comparison)

    missing_reference = pd.DataFrame(
        {
            "cell_id": ["A"] * 4,
            "cycle_index": [3] * 4,
            "voltage_v": reference["voltage_v"],
            "discharge_capacity_ah": reference["discharge_capacity_ah"],
        }
    )
    with pytest.raises(DeltaQError, match="missing reference"):
        compute_delta_q_features(
            missing_reference,
            early_cycles=4,
            reference_cycle=2,
            strict=True,
        )
    with pytest.raises(DeltaQError, match="missing delta-Q columns"):
        compute_delta_q_features(pd.DataFrame({"cell_id": ["A"]}))
    with pytest.raises(ValueError, match="reference_cycle"):
        compute_delta_q_features(missing_reference, early_cycles=2, reference_cycle=3)
    with pytest.raises(ValueError, match="comparison_cycle"):
        compute_delta_q_features(
            missing_reference, early_cycles=4, reference_cycle=2, comparison_cycle=2
        )
    malformed = missing_reference.assign(cycle_index="bad")
    with pytest.raises(DeltaQError, match="non-numeric"):
        compute_delta_q_features(malformed, early_cycles=4, reference_cycle=2)


def test_protocol_parser_and_derived_feature_edge_paths() -> None:
    assert parse_c_rates("3.0C then 1.5c") == (3.0, 1.5)
    stages = protocol_stage_features("2C-1C", max_stages=3)
    assert stages["charge_c_rate_stage_count"] == 2
    assert stages["charge_c_rate_stage_3"] != stages["charge_c_rate_stage_3"]
    empty = protocol_stage_features("CCCV", max_stages=2)
    assert empty["charge_c_rate_stage_count"] == 0
    with pytest.raises(TypeError):
        parse_c_rates(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_stages"):
        protocol_stage_features("1C", max_stages=0)

    series = pd.DataFrame(
        {
            "cell_id": ["A", "A", "A"],
            "cycle_index": [1, 1, 1],
            "sample_index": [0, 1, 2],
            "current_a": [0.0, -1.0, 0.0],
            "voltage_v": [3.0, 3.1, 3.2],
        }
    )
    derived = derive_protocol_features(series)
    assert pd.isna(derived.iloc[0]["charge_current_a_peak"])
    assert pd.isna(derived.iloc[0]["observed_charge_soc_window"])
    with pytest.raises(ValueError, match="missing protocol"):
        derive_protocol_features(series.drop(columns="current_a"))
    with pytest.raises(ValueError, match="nominal_capacity"):
        derive_protocol_features(series, pd.DataFrame({"cell_id": ["A"]}))


def test_trend_and_pipeline_validation_edges() -> None:
    assert pd.isna(linear_trend([1.0], [2.0]))
    assert pd.isna(linear_trend([1.0, 1.0], [2.0, 3.0]))
    assert pd.isna(quadratic_curvature([1.0, 2.0], [1.0, 4.0]))
    assert quadratic_curvature([1.0, 2.0, 3.0], [1.0, 4.0, 9.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="equal length"):
        linear_trend([1.0], [1.0, 2.0])
    for kwargs in (
        {"early_cycles": 1},
        {"early_cycles": 5, "start_cycle": 4, "reference_cycle": 2},
        {"early_cycles": 5, "min_valid_cycles": 6},
    ):
        with pytest.raises(ValueError):
            EarlyCycleFeaturePipeline(**kwargs)

    cells, cycles, timeseries = _feature_tables()
    pipeline = EarlyCycleFeaturePipeline(
        early_cycles=5,
        reference_cycle=2,
        min_valid_cycles=5,
        strict_min_cycles=True,
    )
    with pytest.raises(FeatureExtractionError, match="fewer than"):
        pipeline.transform(cells, cycles.loc[cycles["cycle_index"] <= 3], timeseries)
    with pytest.raises(FeatureExtractionError, match="cells missing"):
        pipeline.transform(cells.drop(columns="nominal_capacity_ah"), cycles, timeseries)
    with pytest.raises(FeatureExtractionError, match="cycles is missing"):
        pipeline.transform(cells, cycles.drop(columns="cycle_index"), timeseries)
