"""Cell-level summary features from an explicitly bounded early window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from batteryguard.features.trends import trend_statistics

_REQUIRED = {
    "cell_id",
    "cycle_index",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "coulombic_efficiency",
    "charge_time_s",
    "discharge_time_s",
}


def _numeric(group: pd.DataFrame, column: str) -> np.ndarray:
    if column not in group.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)


def _finite_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_std": float(np.std(finite, ddof=0)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def summarize_early_cycles(
    cycles: pd.DataFrame,
    *,
    early_cycles: int = 30,
    start_cycle: int = 2,
) -> pd.DataFrame:
    """Build transparent capacity, efficiency, time, thermal, and DCIR features."""

    if early_cycles < start_cycle or start_cycle < 1:
        raise ValueError("require 1 <= start_cycle <= early_cycles")
    missing = _REQUIRED - set(cycles.columns)
    if missing:
        raise ValueError(f"cycles missing required feature columns: {sorted(missing)}")
    numeric_cycle = pd.to_numeric(cycles["cycle_index"], errors="coerce")
    if numeric_cycle.isna().any():
        raise ValueError("cycle_index contains non-numeric values")
    early = cycles.loc[(numeric_cycle >= start_cycle) & (numeric_cycle <= early_cycles)].copy()
    if early.empty:
        return pd.DataFrame(columns=["cell_id", "observed_cycles"])
    early["cycle_index"] = numeric_cycle.loc[early.index].astype(int)
    early.sort_values(["cell_id", "cycle_index"], inplace=True)

    rows: list[dict[str, float | int | str]] = []
    for cell_id, group in early.groupby("cell_id", sort=True):
        cycle_index = _numeric(group, "cycle_index")
        discharge = _numeric(group, "discharge_capacity_ah")
        charge = _numeric(group, "charge_capacity_ah")
        efficiency = _numeric(group, "coulombic_efficiency")
        charge_time = _numeric(group, "charge_time_s")
        discharge_time = _numeric(group, "discharge_time_s")
        row: dict[str, float | int | str] = {
            "cell_id": str(cell_id),
            "observed_cycles": int(pd.Series(cycle_index).nunique()),
            "first_observed_cycle": int(np.min(cycle_index)),
            "last_observed_cycle": int(np.max(cycle_index)),
        }
        for values, prefix in (
            (discharge, "discharge_capacity_ah"),
            (charge, "charge_capacity_ah"),
            (efficiency, "coulombic_efficiency"),
            (charge_time, "charge_time_s"),
            (discharge_time, "discharge_time_s"),
        ):
            row.update(_finite_stats(values, prefix))
            row.update(trend_statistics(cycle_index, values, prefix))

        valid_discharge = discharge[np.isfinite(discharge)]
        if valid_discharge.size:
            reference = float(valid_discharge[0])
            row["capacity_retention"] = (
                float(valid_discharge[-1] / reference) if reference > 0 else float("nan")
            )
            row["capacity_fade_ah"] = float(reference - valid_discharge[-1])
        else:
            row["capacity_retention"] = float("nan")
            row["capacity_fade_ah"] = float("nan")
        valid_efficiency = efficiency[np.isfinite(efficiency)]
        row["efficiency_instability_count"] = int(
            np.sum(valid_efficiency < 0.995)
        )
        row["efficiency_jump_count"] = int(
            np.sum(np.abs(np.diff(valid_efficiency)) > 0.002)
        ) if valid_efficiency.size > 1 else 0

        for optional, prefix in (
            ("dcir_ohm", "dcir_ohm"),
            ("avg_temp_c", "avg_temp_c"),
            ("max_temp_c", "max_temp_c"),
            ("charge_energy_wh", "charge_energy_wh"),
            ("discharge_energy_wh", "discharge_energy_wh"),
        ):
            values = _numeric(group, optional)
            row.update(_finite_stats(values, prefix))
            row.update(trend_statistics(cycle_index, values, prefix))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)
