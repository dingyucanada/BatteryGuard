"""Reproducible cell-level and protocol-holdout split construction."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from batteryguard.schemas.data import SplitName


class SplitBuildError(ValueError):
    """Raised when a leakage-safe split cannot be constructed."""


def _validate_cells(cells: pd.DataFrame) -> pd.DataFrame:
    required = {"cell_id", "protocol_id"}
    missing = required - set(cells.columns)
    if missing:
        raise SplitBuildError(f"cells is missing split columns: {sorted(missing)}")
    if cells.empty:
        raise SplitBuildError("cannot split an empty cells table")
    if cells["cell_id"].isna().any() or cells["cell_id"].astype(str).str.strip().eq("").any():
        raise SplitBuildError("cell_id cannot be missing or blank")
    if cells["cell_id"].duplicated().any():
        duplicates = cells.loc[cells["cell_id"].duplicated(False), "cell_id"].unique().tolist()
        raise SplitBuildError(f"cells must contain one row per cell; duplicates={duplicates[:5]}")
    if cells["protocol_id"].isna().any() or cells["protocol_id"].astype(str).str.strip().eq("").any():
        raise SplitBuildError("protocol_id is required for grouped evaluation")
    return cells.copy()


def _counts(n: int, train_fraction: float, calibration_fraction: float) -> tuple[int, int, int]:
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise SplitBuildError("train_fraction and calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise SplitBuildError("train_fraction + calibration_fraction must be below 1")
    if n < 3:
        raise SplitBuildError("at least three cells are required for train/calibration/test")
    n_train = max(1, int(math.floor(n * train_fraction)))
    n_calibration = max(1, int(math.floor(n * calibration_fraction)))
    if n_train + n_calibration >= n:
        n_train = n - 2
        n_calibration = 1
    return n_train, n_calibration, n - n_train - n_calibration


def _manifest(cells: pd.DataFrame, assignments: dict[str, SplitName], reason: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_id = cells.set_index("cell_id", drop=False)
    for cell_id in cells["cell_id"].astype(str):
        split = assignments[cell_id]
        record = by_id.loc[cell_id]
        group_keys: dict[str, object] = {"protocol_id": str(record["protocol_id"])}
        if "batch_id" in cells.columns and pd.notna(record["batch_id"]):
            group_keys["batch_id"] = str(record["batch_id"])
        rows.append(
            {
                "cell_id": cell_id,
                "split": split.value,
                "split_reason": reason,
                "group_keys": group_keys,
                "ood_type": None,
            }
        )
    return pd.DataFrame(rows)


def build_cell_level_split(
    cells: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Shuffle unique cells, never cycles, into train/calibration/test."""

    checked = _validate_cells(cells)
    n_train, n_calibration, _ = _counts(
        len(checked), train_fraction, calibration_fraction
    )
    rng = np.random.default_rng(seed)
    ids = checked["cell_id"].astype(str).to_numpy(copy=True)
    rng.shuffle(ids)
    assignments: dict[str, SplitName] = {}
    for cell_id in ids[:n_train]:
        assignments[str(cell_id)] = SplitName.TRAIN
    for cell_id in ids[n_train : n_train + n_calibration]:
        assignments[str(cell_id)] = SplitName.CALIBRATION
    for cell_id in ids[n_train + n_calibration :]:
        assignments[str(cell_id)] = SplitName.TEST
    return _manifest(checked, assignments, f"cell-level random seed={seed}")


def build_protocol_holdout_split(
    cells: pd.DataFrame,
    *,
    holdout_protocols: Iterable[str] | None = None,
    calibration_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Reserve complete protocol groups for test, then cell-split calibration.

    Calibration cells are sampled only from non-holdout protocols and are kept
    disjoint from training.  Test protocols can therefore never influence model
    fitting or calibration.
    """

    checked = _validate_cells(cells)
    protocols = sorted(checked["protocol_id"].astype(str).unique())
    if len(protocols) < 2:
        raise SplitBuildError("protocol holdout requires at least two distinct protocols")
    if holdout_protocols is None:
        rng = np.random.default_rng(seed)
        held_out = {str(rng.choice(np.asarray(protocols, dtype=object)))}
    else:
        held_out = {str(value) for value in holdout_protocols}
        unknown = held_out - set(protocols)
        if unknown:
            raise SplitBuildError(f"unknown holdout protocols: {sorted(unknown)}")
        if not held_out or held_out == set(protocols):
            raise SplitBuildError("holdout_protocols must be a non-empty proper subset")

    test_mask = checked["protocol_id"].astype(str).isin(held_out)
    test_ids = checked.loc[test_mask, "cell_id"].astype(str).tolist()
    development_ids = checked.loc[~test_mask, "cell_id"].astype(str).to_numpy(copy=True)
    if len(test_ids) == 0 or len(development_ids) < 2:
        raise SplitBuildError("protocol holdout needs at least one test and two development cells")
    if not 0 < calibration_fraction < 1:
        raise SplitBuildError("calibration_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    rng.shuffle(development_ids)
    n_calibration = max(1, int(math.floor(len(development_ids) * calibration_fraction)))
    if n_calibration >= len(development_ids):
        n_calibration = len(development_ids) - 1
    calibration_ids = {str(value) for value in development_ids[:n_calibration]}
    assignments = {
        str(cell_id): (
            SplitName.TEST
            if str(cell_id) in set(test_ids)
            else SplitName.CALIBRATION
            if str(cell_id) in calibration_ids
            else SplitName.TRAIN
        )
        for cell_id in checked["cell_id"]
    }
    protocol_text = ",".join(sorted(held_out))
    return _manifest(
        checked,
        assignments,
        f"protocol-holdout protocols={protocol_text} seed={seed}",
    )


def build_split_manifest(
    cells: pd.DataFrame,
    *,
    strategy: str = "cell",
    seed: int = 42,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    holdout_protocols: Iterable[str] | None = None,
) -> pd.DataFrame:
    normalized = strategy.lower().replace("_", "-")
    if normalized in {"cell", "cell-level", "grouped-cell"}:
        return build_cell_level_split(
            cells,
            train_fraction=train_fraction,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    if normalized in {"protocol", "protocol-holdout"}:
        return build_protocol_holdout_split(
            cells,
            holdout_protocols=holdout_protocols,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    raise SplitBuildError(f"unknown split strategy: {strategy!r}")
