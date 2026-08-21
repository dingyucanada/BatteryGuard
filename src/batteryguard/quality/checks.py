"""Deterministic canonical-table quality checks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from batteryguard.schemas.data import (
    CellRecord,
    CycleRecord,
    DataQualityIssue,
    DataQualityReport,
    TimeSeriesPoint,
)


class DataQualityError(ValueError):
    """Raised when error-severity quality issues are present."""

    def __init__(self, report: DataQualityReport) -> None:
        self.report = report
        errors = [issue.code for issue in report.issues if issue.severity.upper() == "ERROR"]
        super().__init__(f"data quality gate failed ({len(errors)} errors): {errors[:8]}")


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    cell_id: object | None = None,
    cycle_index: object | None = None,
) -> DataQualityIssue:
    def is_present(value: object | None) -> bool:
        if value is None or value is pd.NA:
            return False
        return not (isinstance(value, float | np.floating) and np.isnan(float(value)))

    cycle: int | None
    try:
        cycle = int(float(str(cycle_index))) if is_present(cycle_index) else None
    except (TypeError, ValueError):
        cycle = None
    return DataQualityIssue(
        code=code,
        severity=severity.upper(),
        message=message,
        cell_id=str(cell_id) if is_present(cell_id) else None,
        cycle_index=cycle,
    )


def _row_values(row: pd.Series, fields: Iterable[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in fields:
        value = row[field]
        if isinstance(value, np.generic):
            value = value.item()
        if value is pd.NA or (
            not isinstance(value, dict | list) and pd.isna(value)
        ):
            value = None
        values[field] = value
    return values


def _schema_issues(
    frame: pd.DataFrame,
    model: type[BaseModel],
    table_name: str,
) -> list[DataQualityIssue]:
    expected = list(model.model_fields)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        return [
            _issue(
                f"{table_name.upper()}_MISSING_COLUMNS",
                "ERROR",
                f"{table_name} missing canonical columns: {missing}",
            )
        ]
    issues: list[DataQualityIssue] = []
    for row_index, row in frame.iterrows():
        try:
            model.model_validate(_row_values(row, expected))
        except ValidationError as exc:
            issues.append(
                _issue(
                    f"{table_name.upper()}_SCHEMA_INVALID",
                    "ERROR",
                    f"row {row_index!r}: {exc.errors(include_url=False)}",
                    cell_id=row.get("cell_id"),
                    cycle_index=row.get("cycle_index"),
                )
            )
    return issues


def run_quality_checks(
    dataset_id: str,
    cells: pd.DataFrame,
    cycles: pd.DataFrame,
    timeseries: pd.DataFrame | None = None,
    *,
    required_early_cycles: int | None = None,
) -> DataQualityReport:
    """Inspect canonical tables and return a report without mutating inputs."""

    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be blank")
    issues: list[DataQualityIssue] = []
    issues.extend(_schema_issues(cells, CellRecord, "cells"))
    issues.extend(_schema_issues(cycles, CycleRecord, "cycles"))
    if timeseries is not None:
        issues.extend(_schema_issues(timeseries, TimeSeriesPoint, "timeseries"))

    if "cell_id" in cells.columns:
        duplicate_cells = cells[cells.duplicated(["cell_id"], keep=False)]
        for cell_id in duplicate_cells["cell_id"].astype(str).unique()[:10]:
            issues.append(
                _issue("DUPLICATE_CELL", "ERROR", "cell_id is not unique", cell_id=cell_id)
            )
    if {"cell_id", "cycle_index"}.issubset(cycles.columns):
        duplicate_cycles = cycles[
            cycles.duplicated(["cell_id", "cycle_index"], keep=False)
        ]
        for _, row in duplicate_cycles.head(10).iterrows():
            issues.append(
                _issue(
                    "DUPLICATE_CYCLE",
                    "ERROR",
                    "(cell_id, cycle_index) is not unique",
                    cell_id=row["cell_id"],
                    cycle_index=row["cycle_index"],
                )
            )
    if timeseries is not None and {
        "cell_id",
        "cycle_index",
        "sample_index",
    }.issubset(timeseries.columns):
        duplicates = timeseries[
            timeseries.duplicated(
                ["cell_id", "cycle_index", "sample_index"], keep=False
            )
        ]
        for _, row in duplicates.head(10).iterrows():
            issues.append(
                _issue(
                    "DUPLICATE_SAMPLE",
                    "ERROR",
                    "time-series primary key is not unique",
                    cell_id=row["cell_id"],
                    cycle_index=row["cycle_index"],
                )
            )

    known_cells = set(cells["cell_id"].dropna().astype(str)) if "cell_id" in cells else set()
    if "cell_id" in cycles:
        for orphan in sorted(set(cycles["cell_id"].dropna().astype(str)) - known_cells)[:10]:
            issues.append(
                _issue("ORPHAN_CYCLE", "ERROR", "cycle references unknown cell", cell_id=orphan)
            )
    if timeseries is not None and "cell_id" in timeseries:
        for orphan in sorted(set(timeseries["cell_id"].dropna().astype(str)) - known_cells)[:10]:
            issues.append(
                _issue(
                    "ORPHAN_TIMESERIES",
                    "ERROR",
                    "time-series references unknown cell",
                    cell_id=orphan,
                )
            )

    if {"cell_id", "cycle_index"}.issubset(cycles.columns):
        for cell_id, group in cycles.groupby("cell_id", sort=False):
            indices = pd.to_numeric(group["cycle_index"], errors="coerce").dropna().astype(int)
            if indices.empty:
                continue
            unique = sorted(indices.unique())
            expected = list(range(unique[0], unique[-1] + 1))
            if unique != expected:
                missing = sorted(set(expected) - set(unique))
                issues.append(
                    _issue(
                        "CYCLE_GAP",
                        "WARNING",
                        f"cycle sequence has gaps; first missing={missing[:5]}",
                        cell_id=cell_id,
                    )
                )
            if required_early_cycles is not None:
                valid_early = sum(1 <= value <= required_early_cycles for value in unique)
                if valid_early < required_early_cycles:
                    issues.append(
                        _issue(
                            "INSUFFICIENT_EARLY_CYCLES",
                            "ERROR",
                            f"requires {required_early_cycles} early cycles, found {valid_early}",
                            cell_id=cell_id,
                        )
                    )

    if timeseries is not None and {
        "cell_id",
        "cycle_index",
        "sample_index",
        "time_s",
    }.issubset(timeseries.columns):
        for (cell_id, cycle_index), group in timeseries.groupby(
            ["cell_id", "cycle_index"], sort=False
        ):
            ordered = group.sort_values("sample_index")
            sample = pd.to_numeric(ordered["sample_index"], errors="coerce").to_numpy()
            time = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy()
            if np.any(~np.isfinite(sample)) or np.any(np.diff(sample) <= 0):
                issues.append(
                    _issue(
                        "SAMPLE_ORDER_INVALID",
                        "ERROR",
                        "sample_index must be finite and strictly increasing",
                        cell_id=cell_id,
                        cycle_index=cycle_index,
                    )
                )
            if np.any(~np.isfinite(time)) or np.any(np.diff(time) < 0):
                issues.append(
                    _issue(
                        "TIME_ORDER_INVALID",
                        "ERROR",
                        "time_s must be finite and non-decreasing within a cycle",
                        cell_id=cell_id,
                        cycle_index=cycle_index,
                    )
                )

    error_count = sum(issue.severity == "ERROR" for issue in issues)
    warning_count = sum(issue.severity == "WARNING" for issue in issues)
    denominator = max(1, len(cells) + len(cycles))
    quality_score = float(
        np.clip(1.0 - (error_count * 2.0 + warning_count * 0.25) / denominator, 0.0, 1.0)
    )

    invalid_cells = {issue.cell_id for issue in issues if issue.severity == "ERROR" and issue.cell_id}
    invalid_cycle_keys = {
        (issue.cell_id, issue.cycle_index)
        for issue in issues
        if issue.severity == "ERROR" and issue.cell_id and issue.cycle_index is not None
    }
    valid_cells = max(0, len(cells) - len(invalid_cells))
    valid_cycles = len(cycles)
    if {"cell_id", "cycle_index"}.issubset(cycles.columns):
        invalid_count = 0
        for _, row in cycles.iterrows():
            cell_id = str(row["cell_id"])
            try:
                cycle_index = int(row["cycle_index"])
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            if (cell_id, cycle_index) in invalid_cycle_keys or cell_id in invalid_cells:
                invalid_count += 1
        valid_cycles -= invalid_count
    return DataQualityReport(
        dataset_id=dataset_id,
        quality_score=quality_score,
        valid_cells=valid_cells,
        valid_cycles=max(0, valid_cycles),
        issues=issues,
        leakage_checks_passed=False,
    )


def assert_data_quality(
    dataset_id: str,
    cells: pd.DataFrame,
    cycles: pd.DataFrame,
    timeseries: pd.DataFrame | None = None,
    *,
    required_early_cycles: int | None = None,
) -> DataQualityReport:
    """Run quality checks and hard-fail if any error is found."""

    report = run_quality_checks(
        dataset_id,
        cells,
        cycles,
        timeseries,
        required_early_cycles=required_early_cycles,
    )
    if any(issue.severity == "ERROR" for issue in report.issues):
        raise DataQualityError(report)
    return report
