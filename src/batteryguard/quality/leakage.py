"""Hard leakage invariants for splits, features, and model fitting scopes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd


class DataLeakageError(RuntimeError):
    """Raised whenever an invariant could expose a hidden lifetime label."""


_FORBIDDEN_EXACT = {
    "cycle_life",
    "cell_life",
    "lifetime",
    "full_lifetime",
    "full_cycle_life",
    "eol_cycle",
    "end_of_life_cycle",
    "remaining_useful_life",
    "rul",
    "target",
    "label",
    "protocol_id",
    "policy_id",
    "filename",
    "file_name",
    "file_path",
    "source_file",
    "source_path",
}
_FORBIDDEN_PARTS = ("future_", "post_eol", "full_life", "actual_life", "hidden_life")


def assert_cell_level_split(splits: pd.DataFrame) -> None:
    required = {"cell_id", "split"}
    missing = required - set(splits.columns)
    if missing:
        raise DataLeakageError(f"split manifest missing columns: {sorted(missing)}")
    if splits.empty:
        raise DataLeakageError("split manifest is empty")
    normalized = splits.assign(
        cell_id=splits["cell_id"].astype(str), split=splits["split"].astype(str)
    )
    counts = normalized.groupby("cell_id")["split"].nunique()
    leaked = counts[counts > 1].index.tolist()
    duplicates = normalized[normalized.duplicated("cell_id", keep=False)]["cell_id"].unique().tolist()
    if leaked or duplicates:
        raise DataLeakageError(
            f"cell-level split violation; cells occur multiple times/splits: {sorted(set(leaked + duplicates))[:10]}"
        )
    allowed = {"train", "calibration", "test", "demo_hidden", "external_ood"}
    unknown = set(normalized["split"]) - allowed
    if unknown:
        raise DataLeakageError(f"unknown split names: {sorted(unknown)}")


def assert_protocol_holdout(cells: pd.DataFrame, splits: pd.DataFrame) -> None:
    assert_cell_level_split(splits)
    required = {"cell_id", "protocol_id"}
    missing = required - set(cells.columns)
    if missing:
        raise DataLeakageError(f"cells missing protocol holdout columns: {sorted(missing)}")
    merged = splits[["cell_id", "split"]].merge(
        cells[["cell_id", "protocol_id"]], on="cell_id", how="left", validate="one_to_one"
    )
    if merged["protocol_id"].isna().any():
        missing_ids = merged.loc[merged["protocol_id"].isna(), "cell_id"].tolist()
        raise DataLeakageError(f"split cells missing protocol metadata: {missing_ids[:10]}")
    development = set(
        merged.loc[merged["split"].isin(["train", "calibration"]), "protocol_id"].astype(str)
    )
    test = set(merged.loc[merged["split"] == "test", "protocol_id"].astype(str))
    overlap = sorted(development & test)
    if overlap:
        raise DataLeakageError(f"test protocols overlap development protocols: {overlap}")


def assert_no_future_cycles(cycles: pd.DataFrame, early_cycles: int) -> None:
    if early_cycles < 1:
        raise ValueError("early_cycles must be positive")
    if "cycle_index" not in cycles.columns:
        raise DataLeakageError("cycle source is missing cycle_index")
    numeric = pd.to_numeric(cycles["cycle_index"], errors="coerce")
    if numeric.isna().any():
        raise DataLeakageError("cycle_index contains non-numeric values")
    future = cycles.loc[numeric > early_cycles, [column for column in ["cell_id", "cycle_index"] if column in cycles]]
    if not future.empty:
        raise DataLeakageError(
            f"feature source references cycles after early_cycles={early_cycles}: "
            f"{future.head(5).to_dict(orient='records')}"
        )


def _normalized_column(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")


def assert_feature_frame_safe(features: pd.DataFrame, *, early_cycles: int | None = None) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    violations: list[str] = []
    for column in features.columns:
        normalized = _normalized_column(column)
        if normalized in _FORBIDDEN_EXACT or any(part in normalized for part in _FORBIDDEN_PARTS):
            violations.append(str(column))
            continue
        if early_cycles is not None:
            cycle_match = re.search(r"(?:cycle|c)_?(\d+)", normalized)
            if cycle_match and int(cycle_match.group(1)) > early_cycles:
                violations.append(str(column))
    if violations:
        raise DataLeakageError(f"forbidden/leaky feature columns: {sorted(set(violations))}")
    if "cell_id" in features.columns and features["cell_id"].duplicated().any():
        raise DataLeakageError("feature frame must contain at most one row per cell")


def assert_training_scope(training_cell_ids: Iterable[str], splits: pd.DataFrame) -> None:
    assert_cell_level_split(splits)
    split_by_id = {
        str(row["cell_id"]): str(row["split"])
        for _, row in splits[["cell_id", "split"]].iterrows()
    }
    ids = {str(value) for value in training_cell_ids}
    unknown = sorted(ids - set(split_by_id))
    invalid = sorted(cell_id for cell_id in ids if split_by_id.get(cell_id) != "train")
    if unknown:
        raise DataLeakageError(f"training contains cells absent from manifest: {unknown[:10]}")
    if invalid:
        details = {cell_id: split_by_id[cell_id] for cell_id in invalid[:10]}
        raise DataLeakageError(f"only train split may fit a model; invalid={details}")


def assert_split_isolation(split_members: Mapping[str, Iterable[str]]) -> None:
    normalized = {name: {str(value) for value in values} for name, values in split_members.items()}
    names = sorted(normalized)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(normalized[left] & normalized[right])
            if overlap:
                raise DataLeakageError(f"{left}/{right} cell overlap: {overlap[:10]}")


def assert_no_leakage(
    cells: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    features: pd.DataFrame | None = None,
    feature_source_cycles: pd.DataFrame | None = None,
    early_cycles: int | None = None,
    require_protocol_holdout: bool = False,
    training_cell_ids: Iterable[str] | None = None,
) -> None:
    """Apply all relevant leakage gates as one hard-failing boundary."""

    assert_cell_level_split(splits)
    cell_ids = set(cells["cell_id"].astype(str)) if "cell_id" in cells else set()
    split_ids = set(splits["cell_id"].astype(str))
    if cell_ids != split_ids:
        raise DataLeakageError(
            f"split coverage differs from cells (missing={sorted(cell_ids - split_ids)[:5]}, "
            f"extra={sorted(split_ids - cell_ids)[:5]})"
        )
    if require_protocol_holdout:
        assert_protocol_holdout(cells, splits)
    if features is not None:
        assert_feature_frame_safe(features, early_cycles=early_cycles)
    if feature_source_cycles is not None:
        if early_cycles is None:
            raise ValueError("early_cycles is required with feature_source_cycles")
        assert_no_future_cycles(feature_source_cycles, early_cycles)
    if training_cell_ids is not None:
        assert_training_scope(training_cell_ids, splits)
