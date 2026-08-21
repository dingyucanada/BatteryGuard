"""Read and write standardized CSV/Parquet datasets.

The adapter deliberately validates every canonical row with the public Pydantic
schemas.  It therefore fails at the ingestion boundary instead of allowing an
invalid unit, fabricated censored label, or duplicate primary key to reach a
model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from batteryguard.schemas.data import CellRecord, CycleRecord, SplitRecord, TimeSeriesPoint


class DataIngestionError(ValueError):
    """Raised when source data cannot satisfy the canonical data contract."""


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _empty_frame(model: type[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(model.model_fields))


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or (not isinstance(value, list | dict) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def _decode_group_keys(value: Any) -> dict[str, Any]:
    value = _python_value(value)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DataIngestionError(f"invalid JSON in splits.group_keys: {value!r}") from exc
        if not isinstance(decoded, dict):
            raise DataIngestionError("splits.group_keys JSON must decode to an object")
        return decoded
    raise DataIngestionError("splits.group_keys must be a mapping or JSON object string")


def validate_frame(
    frame: pd.DataFrame,
    model: type[_ModelT],
    *,
    table_name: str,
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    """Return a canonicalized frame or raise with the offending row identified."""

    if not isinstance(frame, pd.DataFrame):
        raise DataIngestionError(f"{table_name} must be a pandas DataFrame")
    expected = list(model.model_fields)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise DataIngestionError(f"{table_name} is missing required contract columns: {missing}")
    extra = [str(column) for column in frame.columns if column not in expected]
    if extra and not allow_extra_columns:
        raise DataIngestionError(f"{table_name} has unexpected columns: {extra}")

    canonical: list[dict[str, Any]] = []
    for row_index, row in frame.loc[:, expected].iterrows():
        values = {name: _python_value(row[name]) for name in expected}
        if model is SplitRecord:
            values["group_keys"] = _decode_group_keys(values["group_keys"])
        try:
            canonical.append(model.model_validate(values).model_dump(mode="python"))
        except ValidationError as exc:
            raise DataIngestionError(
                f"{table_name} row {row_index!r} violates the canonical schema: {exc}"
            ) from exc
    return pd.DataFrame(canonical, columns=expected)


def _assert_unique(frame: pd.DataFrame, keys: list[str], table_name: str) -> None:
    if frame.empty:
        return
    duplicate_mask = frame.duplicated(keys, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, keys].head(5).to_dict(orient="records")
        raise DataIngestionError(
            f"{table_name} violates primary key {keys}; duplicate examples: {examples}"
        )


@dataclass(slots=True)
class CanonicalDataset:
    """In-memory canonical dataset used by downstream work packages."""

    cells: pd.DataFrame
    cycles: pd.DataFrame
    timeseries: pd.DataFrame
    splits: pd.DataFrame | None = None
    dataset_id: str = "dataset"

    def validate(self) -> CanonicalDataset:
        self.cells = validate_frame(self.cells, CellRecord, table_name="cells")
        self.cycles = validate_frame(self.cycles, CycleRecord, table_name="cycles")
        self.timeseries = validate_frame(
            self.timeseries, TimeSeriesPoint, table_name="timeseries"
        )
        if self.splits is not None:
            self.splits = validate_frame(self.splits, SplitRecord, table_name="splits")

        _assert_unique(self.cells, ["cell_id"], "cells")
        _assert_unique(self.cycles, ["cell_id", "cycle_index"], "cycles")
        _assert_unique(
            self.timeseries,
            ["cell_id", "cycle_index", "sample_index"],
            "timeseries",
        )
        if self.splits is not None:
            _assert_unique(self.splits, ["cell_id"], "splits")

        known_cells = set(self.cells["cell_id"].astype(str))
        cycle_cells = set(self.cycles["cell_id"].astype(str))
        series_cells = set(self.timeseries["cell_id"].astype(str))
        orphan_cycles = sorted(cycle_cells - known_cells)
        orphan_series = sorted(series_cells - known_cells)
        if orphan_cycles:
            raise DataIngestionError(f"cycles references unknown cells: {orphan_cycles[:5]}")
        if orphan_series:
            raise DataIngestionError(f"timeseries references unknown cells: {orphan_series[:5]}")
        if self.splits is not None:
            split_cells = set(self.splits["cell_id"].astype(str))
            if split_cells != known_cells:
                missing = sorted(known_cells - split_cells)
                extra = sorted(split_cells - known_cells)
                raise DataIngestionError(
                    f"splits must cover cells exactly (missing={missing[:5]}, extra={extra[:5]})"
                )
        return self


def load_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise DataIngestionError(f"table does not exist: {source}")
    suffix = source.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(source)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(source)
    except Exception as exc:
        raise DataIngestionError(f"failed to read {source}: {exc}") from exc
    raise DataIngestionError(f"unsupported table format {suffix!r}; expected CSV or Parquet")


def _discover_table(directory: Path, name: str, *, required: bool) -> Path | None:
    matches = [directory / f"{name}.parquet", directory / f"{name}.pq", directory / f"{name}.csv"]
    existing = [path for path in matches if path.is_file()]
    if len(existing) > 1:
        raise DataIngestionError(
            f"ambiguous {name} table in {directory}; keep only one CSV/Parquet representation"
        )
    if existing:
        return existing[0]
    if required:
        raise DataIngestionError(f"missing {name}.csv or {name}.parquet in {directory}")
    return None


def load_standardized(
    source: str | Path | Mapping[str, str | Path], *, dataset_id: str | None = None
) -> CanonicalDataset:
    """Load a canonical dataset from a directory or explicit table mapping."""

    paths: dict[str, Path | None]
    if isinstance(source, Mapping):
        missing = {"cells", "cycles", "timeseries"} - set(source)
        if missing:
            raise DataIngestionError(f"table mapping is missing keys: {sorted(missing)}")
        paths = {
            name: Path(source[name]) if name in source and source[name] is not None else None
            for name in ("cells", "cycles", "timeseries", "splits")
        }
        inferred_id = "dataset"
    else:
        directory = Path(source)
        if not directory.is_dir():
            raise DataIngestionError(f"standardized source must be a directory: {directory}")
        paths = {
            "cells": _discover_table(directory, "cells", required=True),
            "cycles": _discover_table(directory, "cycles", required=True),
            "timeseries": _discover_table(directory, "timeseries", required=True),
            "splits": _discover_table(directory, "splits", required=False),
        }
        inferred_id = directory.name

    return CanonicalDataset(
        cells=load_table(paths["cells"]),  # type: ignore[arg-type]
        cycles=load_table(paths["cycles"]),  # type: ignore[arg-type]
        timeseries=load_table(paths["timeseries"]),  # type: ignore[arg-type]
        splits=load_table(paths["splits"]) if paths["splits"] is not None else None,
        dataset_id=dataset_id or inferred_id,
    ).validate()


def write_standardized(
    dataset: CanonicalDataset,
    destination: str | Path,
    *,
    file_format: str = "parquet",
) -> dict[str, Path]:
    """Validate and atomically-shaped-write canonical tables.

    Individual file replacement is delegated to pandas/pyarrow; callers should
    write to a staging directory when they need multi-table transactionality.
    """

    validated = dataset.validate()
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    normalized_format = file_format.lower().lstrip(".")
    if normalized_format not in {"csv", "parquet"}:
        raise DataIngestionError("file_format must be 'csv' or 'parquet'")

    tables: dict[str, pd.DataFrame | None] = {
        "cells": validated.cells,
        "cycles": validated.cycles,
        "timeseries": validated.timeseries,
        "splits": validated.splits,
    }
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        if frame is None:
            continue
        path = output / f"{name}.{normalized_format}"
        serialized = frame.copy()
        if name == "splits" and normalized_format == "csv":
            serialized["group_keys"] = serialized["group_keys"].map(
                lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
            )
        try:
            if normalized_format == "csv":
                serialized.to_csv(path, index=False)
            else:
                serialized.to_parquet(path, index=False)
        except Exception as exc:
            raise DataIngestionError(f"failed to write {path}: {exc}") from exc
        written[name] = path
    return written


def empty_timeseries() -> pd.DataFrame:
    """Return a correctly shaped empty time-series table for summary-only sources."""

    return _empty_frame(TimeSeriesPoint)
