from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from batteryguard.ingestion import (
    AdapterRegistry,
    CanonicalDataset,
    DataIngestionError,
    MATRAdapter,
    SplitBuildError,
    build_cell_level_split,
    build_protocol_holdout_split,
    build_split_manifest,
    load_matr,
    load_standardized,
    write_standardized,
)
from batteryguard.ingestion.standard import load_table, validate_frame
from batteryguard.schemas.data import CellRecord, CycleRecord, TimeSeriesPoint


def _canonical() -> CanonicalDataset:
    cells = pd.DataFrame(
        [
            {
                "cell_id": f"C{index}",
                "chemistry": "LFP_graphite",
                "nominal_capacity_ah": 1.1,
                "batch_id": "B1",
                "protocol_id": f"P{index % 2}",
                "cycle_life": 500 + index * 10,
                "eol_threshold": 0.8,
                "censored": False,
            }
            for index in range(6)
        ],
        columns=list(CellRecord.model_fields),
    )
    cycles = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "cycle_index": 1,
                "charge_capacity_ah": 1.1,
                "discharge_capacity_ah": 1.08,
                "coulombic_efficiency": 0.982,
                "charge_time_s": 3600.0,
                "discharge_time_s": 3500.0,
                "dcir_ohm": 0.02,
                "avg_temp_c": 25.0,
                "max_temp_c": 28.0,
                "charge_energy_wh": None,
                "discharge_energy_wh": None,
            }
            for cell_id in cells["cell_id"]
        ],
        columns=list(CycleRecord.model_fields),
    )
    timeseries = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "cycle_index": 1,
                "sample_index": 0,
                "time_s": 0.0,
                "current_a": -1.0,
                "voltage_v": 3.3,
                "charge_capacity_ah": 0.0,
                "discharge_capacity_ah": 0.0,
                "temperature_c": 25.0,
                "step_type": "discharge",
            }
            for cell_id in cells["cell_id"]
        ],
        columns=list(TimeSeriesPoint.model_fields),
    )
    return CanonicalDataset(cells, cycles, timeseries, dataset_id="fixture")


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_standardized_round_trip(tmp_path, file_format: str) -> None:
    dataset = _canonical()
    dataset.splits = build_cell_level_split(dataset.cells, seed=7)
    write_standardized(dataset, tmp_path, file_format=file_format)
    loaded = load_standardized(tmp_path)
    assert len(loaded.cells) == 6
    assert loaded.splits is not None
    assert len(loaded.splits) == 6
    assert isinstance(loaded.splits.iloc[0]["group_keys"], dict)


def test_cell_and_protocol_splits_are_reproducible_and_disjoint() -> None:
    cells = _canonical().cells
    first = build_cell_level_split(cells, seed=11)
    second = build_cell_level_split(cells, seed=11)
    pd.testing.assert_frame_equal(first, second)
    protocol = build_protocol_holdout_split(cells, holdout_protocols=["P1"], seed=2)
    test_ids = set(protocol.loc[protocol["split"] == "test", "cell_id"])
    assert test_ids == set(cells.loc[cells["protocol_id"] == "P1", "cell_id"])


def test_matr_classic_struct_adapter(tmp_path) -> None:
    raw_cycles = []
    for _ in range(3):
        raw_cycles.append(
            {
                "t": np.arange(8.0),
                "I": np.asarray([1, 1, 1, 0, -1, -1, -1, -1], dtype=float),
                "V": np.linspace(2.8, 3.5, 8),
                "Qc": np.asarray([0, 0.3, 0.7, 1, 1, 1, 1, 1], dtype=float),
                "Qd": np.asarray([0, 0, 0, 0, 0.2, 0.5, 0.8, 0.99], dtype=float),
                "T": np.full(8, 25.0),
            }
        )
    batch = {
        "barcode": "severson-test",
        "policy_readable": "3C",
        "cell_life": 900,
        "summary": {
            "cycle": np.arange(1, 4),
            "QCharge": np.ones(3),
            "QDischarge": np.full(3, 0.99),
            "CE": np.full(3, 0.99),
            "chargetime": np.full(3, 3.0),
        },
        "cycles": np.asarray(raw_cycles, dtype=object),
    }
    source = tmp_path / "batch.mat"
    savemat(source, {"batch": batch})
    dataset = load_matr(source, source_time_unit="seconds")
    assert dataset.cells.iloc[0]["cell_id"] == "severson-test"
    assert dataset.cells.iloc[0]["cycle_life"] == 900
    assert len(dataset.cycles) == 3
    assert len(dataset.timeseries) == 24
    assert dataset.cycles["discharge_time_s"].gt(0).all()


def test_standard_adapter_rejects_missing_contract_column(tmp_path) -> None:
    dataset = _canonical()
    dataset.cells.drop(columns="chemistry").to_csv(tmp_path / "cells.csv", index=False)
    dataset.cycles.to_csv(tmp_path / "cycles.csv", index=False)
    dataset.timeseries.to_csv(tmp_path / "timeseries.csv", index=False)
    with pytest.raises(DataIngestionError, match="missing required contract columns"):
        load_standardized(tmp_path)


def test_standard_mapping_and_contract_failure_paths(tmp_path) -> None:
    dataset = _canonical()
    paths = write_standardized(dataset, tmp_path, file_format="csv")
    loaded = load_standardized(paths, dataset_id="mapped")
    assert loaded.dataset_id == "mapped"

    with pytest.raises(DataIngestionError, match="missing keys"):
        load_standardized({"cells": paths["cells"]})
    with pytest.raises(DataIngestionError, match="must be a directory"):
        load_standardized(tmp_path / "absent")
    with pytest.raises(DataIngestionError, match="does not exist"):
        load_table(tmp_path / "absent.csv")
    text = tmp_path / "table.txt"
    text.write_text("not a table", encoding="utf-8")
    with pytest.raises(DataIngestionError, match="unsupported table format"):
        load_table(text)
    with pytest.raises(DataIngestionError, match="file_format"):
        write_standardized(dataset, tmp_path / "bad", file_format="json")

    extra = dataset.cells.assign(unexpected=1)
    with pytest.raises(DataIngestionError, match="unexpected columns"):
        validate_frame(extra, CellRecord, table_name="cells")
    assert "unexpected" not in validate_frame(
        extra, CellRecord, table_name="cells", allow_extra_columns=True
    )


def test_canonical_dataset_rejects_orphans_duplicates_and_split_gaps() -> None:
    dataset = _canonical().validate()
    duplicate_cycles = pd.concat([dataset.cycles, dataset.cycles.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataIngestionError, match="primary key"):
        CanonicalDataset(dataset.cells.copy(), duplicate_cycles, dataset.timeseries.copy()).validate()

    orphan_cycles = dataset.cycles.copy()
    orphan_cycles.loc[0, "cell_id"] = "UNKNOWN"
    with pytest.raises(DataIngestionError, match="unknown cells"):
        CanonicalDataset(dataset.cells.copy(), orphan_cycles, dataset.timeseries.copy()).validate()

    orphan_series = dataset.timeseries.copy()
    orphan_series.loc[0, "cell_id"] = "UNKNOWN"
    with pytest.raises(DataIngestionError, match="timeseries references"):
        CanonicalDataset(dataset.cells.copy(), dataset.cycles.copy(), orphan_series).validate()

    incomplete = build_cell_level_split(dataset.cells).iloc[:-1].copy()
    with pytest.raises(DataIngestionError, match="cover cells exactly"):
        CanonicalDataset(
            dataset.cells.copy(), dataset.cycles.copy(), dataset.timeseries.copy(), incomplete
        ).validate()


def test_split_builder_rejects_invalid_inputs_and_dispatches_protocol() -> None:
    cells = _canonical().cells
    invalid_frames = [
        cells.drop(columns="protocol_id"),
        cells.iloc[0:0],
        cells.assign(cell_id=[None, "C1", "C2", "C3", "C4", "C5"]),
        pd.concat([cells, cells.iloc[[0]]], ignore_index=True),
        cells.assign(protocol_id=""),
    ]
    for frame in invalid_frames:
        with pytest.raises(SplitBuildError):
            build_cell_level_split(frame)
    with pytest.raises(SplitBuildError, match="train_fraction"):
        build_cell_level_split(cells, train_fraction=0)
    with pytest.raises(SplitBuildError, match="below 1"):
        build_cell_level_split(cells, train_fraction=0.8, calibration_fraction=0.3)
    with pytest.raises(SplitBuildError, match="at least three"):
        build_cell_level_split(cells.iloc[:2])

    automatic = build_protocol_holdout_split(cells, seed=4)
    assert set(automatic["split"]) == {"train", "calibration", "test"}
    with pytest.raises(SplitBuildError, match="unknown holdout"):
        build_protocol_holdout_split(cells, holdout_protocols=["UNKNOWN"])
    with pytest.raises(SplitBuildError, match="proper subset"):
        build_protocol_holdout_split(cells, holdout_protocols=[])
    with pytest.raises(SplitBuildError, match="at least two"):
        build_protocol_holdout_split(cells.assign(protocol_id="one"))
    protocol = build_split_manifest(
        cells, strategy="protocol_holdout", holdout_protocols=["P1"], seed=9
    )
    assert set(protocol.loc[protocol["split"] == "test", "cell_id"]) == {"C1", "C3", "C5"}
    with pytest.raises(SplitBuildError, match="unknown split strategy"):
        build_split_manifest(cells, strategy="row-random")


def test_adapter_registry_and_matr_summary_only_paths(tmp_path) -> None:
    registry = AdapterRegistry()
    registry.register("fixture", lambda source, **_: _canonical())
    assert registry.names() == ("fixture",)
    assert registry.load("fixture", tmp_path).dataset_id == "fixture"
    with pytest.raises(ValueError, match="blank"):
        registry.register(" ", lambda source, **_: _canonical())
    with pytest.raises(ValueError, match="already"):
        registry.register("fixture", lambda source, **_: _canonical())
    with pytest.raises(DataIngestionError, match="unknown adapter"):
        registry.load("missing", tmp_path)

    batch = {
        "barcode": "summary-only",
        "policy_readable": "1C",
        "cell_life": np.nan,
        "summary": {
            "cycle": np.asarray([1, 2]),
            "QCharge": np.asarray([1.0, 0.99]),
            "QDischarge": np.asarray([0.99, 0.98]),
            "CE": np.asarray([0.99, 0.989]),
            "charge_time_s": np.asarray([3600.0, 3601.0]),
            "discharge_time_s": np.asarray([3500.0, 3501.0]),
        },
    }
    source = tmp_path / "summary.mat"
    savemat(source, {"batch": batch})
    loaded = MATRAdapter(source_time_unit="seconds").load(source)
    assert loaded.cells.iloc[0]["censored"]
    assert pd.isna(loaded.cells.iloc[0]["cycle_life"])
    assert loaded.timeseries.empty

    with pytest.raises(ValueError, match="nominal"):
        MATRAdapter(nominal_capacity_ah=0)
    with pytest.raises(ValueError, match="eol_threshold"):
        MATRAdapter(eol_threshold=1.0)
    with pytest.raises(DataIngestionError, match="does not exist"):
        MATRAdapter().load(tmp_path / "missing.mat")
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("x", encoding="utf-8")
    with pytest.raises(DataIngestionError, match="expects a .mat"):
        MATRAdapter().load(wrong)
    no_batch = tmp_path / "no-batch.mat"
    savemat(no_batch, {"other": np.asarray([1.0])})
    with pytest.raises(DataIngestionError, match="no 'batch'"):
        MATRAdapter().load(no_batch)
