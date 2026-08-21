"""Voltage-aligned delta-Q(V) features from two early discharge curves."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class DeltaQError(ValueError):
    """Raised for malformed curve data or incompatible requested cycles."""


def _discharge_curve(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    voltage = pd.to_numeric(group["voltage_v"], errors="coerce").to_numpy(dtype=float)
    capacity = pd.to_numeric(
        group["discharge_capacity_ah"], errors="coerce"
    ).to_numpy(dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(capacity)
    if "step_type" in group.columns:
        step = group["step_type"].astype("string").str.lower().to_numpy()
        discharge_mask = np.asarray(step == "discharge", dtype=bool)
        if np.sum(mask & discharge_mask) >= 3:
            mask &= discharge_mask
    elif "current_a" in group.columns:
        current = pd.to_numeric(group["current_a"], errors="coerce").to_numpy(dtype=float)
        discharge_mask = current < -1e-8
        if np.sum(mask & discharge_mask) >= 3:
            mask &= discharge_mask
    voltage = voltage[mask]
    capacity = capacity[mask]
    if voltage.size < 3:
        raise DeltaQError("a discharge curve requires at least three finite points")
    order = np.argsort(voltage)
    curve = pd.DataFrame({"voltage": voltage[order], "capacity": capacity[order]})
    curve = curve.groupby("voltage", as_index=False, sort=True).agg(
        capacity=("capacity", "mean")
    )
    if len(curve) < 3 or float(curve["voltage"].max() - curve["voltage"].min()) <= 0:
        raise DeltaQError("discharge curve has insufficient distinct voltage support")
    return curve["voltage"].to_numpy(), curve["capacity"].to_numpy()


def delta_q_curve(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    voltage_grid: Iterable[float] | None = None,
    grid_points: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return common voltage grid and Q_comparison(V) - Q_reference(V)."""

    if grid_points < 8:
        raise ValueError("grid_points must be at least 8")
    ref_voltage, ref_capacity = _discharge_curve(reference)
    cmp_voltage, cmp_capacity = _discharge_curve(comparison)
    low = max(float(np.min(ref_voltage)), float(np.min(cmp_voltage)))
    high = min(float(np.max(ref_voltage)), float(np.max(cmp_voltage)))
    if high <= low:
        raise DeltaQError("reference and comparison curves have no overlapping voltage range")
    if voltage_grid is None:
        grid = np.linspace(low, high, grid_points)
    else:
        grid = np.asarray(list(voltage_grid), dtype=float).reshape(-1)
        if grid.size < 3 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise DeltaQError("voltage_grid must contain at least three increasing finite values")
        if float(grid[0]) < low or float(grid[-1]) > high:
            raise DeltaQError("voltage_grid extends outside common curve support")
    reference_q = np.interp(grid, ref_voltage, ref_capacity)
    comparison_q = np.interp(grid, cmp_voltage, cmp_capacity)
    return grid, comparison_q - reference_q


def compute_delta_q_features(
    timeseries: pd.DataFrame,
    *,
    early_cycles: int = 30,
    reference_cycle: int = 2,
    comparison_cycle: int | None = None,
    grid_points: int = 128,
    strict: bool = False,
) -> pd.DataFrame:
    """Compute voltage-aligned curve-shift summaries for every possible cell."""

    required = {"cell_id", "cycle_index", "voltage_v", "discharge_capacity_ah"}
    missing = required - set(timeseries.columns)
    if missing:
        raise DeltaQError(f"timeseries missing delta-Q columns: {sorted(missing)}")
    if reference_cycle < 1 or reference_cycle > early_cycles:
        raise ValueError("reference_cycle must lie inside the early-cycle window")
    requested_comparison = comparison_cycle or early_cycles
    if requested_comparison <= reference_cycle or requested_comparison > early_cycles:
        raise ValueError("comparison_cycle must be after reference_cycle and within early_cycles")

    numeric_cycle = pd.to_numeric(timeseries["cycle_index"], errors="coerce")
    if numeric_cycle.isna().any():
        raise DeltaQError("cycle_index contains non-numeric values")
    bounded = timeseries.loc[numeric_cycle <= early_cycles].copy()
    bounded["cycle_index"] = numeric_cycle.loc[bounded.index].astype(int)
    rows: list[dict[str, float | int | str]] = []
    failures: list[str] = []
    for cell_id, group in bounded.groupby("cell_id", sort=True):
        available = sorted(group["cycle_index"].unique())
        if reference_cycle not in available:
            failures.append(f"{cell_id}: missing reference cycle {reference_cycle}")
            continue
        candidates = [value for value in available if reference_cycle < value <= requested_comparison]
        if not candidates:
            failures.append(f"{cell_id}: no comparison cycle after {reference_cycle}")
            continue
        selected = requested_comparison if requested_comparison in candidates else max(candidates)
        try:
            grid, delta = delta_q_curve(
                group[group["cycle_index"] == reference_cycle],
                group[group["cycle_index"] == selected],
                grid_points=grid_points,
            )
        except DeltaQError as exc:
            failures.append(f"{cell_id}: {exc}")
            continue
        variance = float(np.var(delta, ddof=0))
        rows.append(
            {
                "cell_id": str(cell_id),
                "delta_q_reference_cycle": reference_cycle,
                "delta_q_comparison_cycle": int(selected),
                "delta_q_mean_ah": float(np.mean(delta)),
                "delta_q_std_ah": float(np.std(delta, ddof=0)),
                "delta_q_min_ah": float(np.min(delta)),
                "delta_q_max_ah": float(np.max(delta)),
                "delta_q_range_ah": float(np.ptp(delta)),
                "delta_q_l1_ah": float(np.mean(np.abs(delta))),
                "delta_q_l2_ah": float(np.sqrt(np.mean(np.square(delta)))),
                "delta_q_log_variance": float(np.log10(max(variance, np.finfo(float).tiny))),
                "delta_q_voltage_overlap_v": float(grid[-1] - grid[0]),
            }
        )
    if strict and failures:
        raise DeltaQError("delta-Q extraction failed: " + "; ".join(failures[:8]))
    return pd.DataFrame(rows)
