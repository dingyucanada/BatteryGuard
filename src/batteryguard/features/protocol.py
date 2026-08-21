"""Physical charging-protocol features without using protocol identifiers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def parse_c_rates(protocol_text: str) -> tuple[float, ...]:
    """Extract explicit values followed by C/c from readable policy text."""

    if not isinstance(protocol_text, str):
        raise TypeError("protocol_text must be a string")
    values = [float(match) for match in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[Cc]\b", protocol_text)]
    return tuple(values)


def protocol_stage_features(protocol_text: str, *, max_stages: int = 6) -> dict[str, float]:
    """Convert readable C-rates to numeric stage features.

    Callers must never include the original policy string or protocol ID in a
    model frame.  The main pipeline derives equivalent values from measured
    current and therefore does not call this parser by default.
    """

    if max_stages < 1:
        raise ValueError("max_stages must be positive")
    rates = parse_c_rates(protocol_text)
    result = {
        f"charge_c_rate_stage_{index + 1}": rates[index] if index < len(rates) else float("nan")
        for index in range(max_stages)
    }
    finite = np.asarray(rates, dtype=float)
    result["charge_c_rate_stage_count"] = float(len(rates))
    result["charge_c_rate_peak"] = float(np.max(finite)) if finite.size else float("nan")
    result["charge_c_rate_mean"] = float(np.mean(finite)) if finite.size else float("nan")
    return result


def _stage_count(active: np.ndarray) -> int:
    if active.size == 0:
        return 0
    return int(np.sum(active & ~np.concatenate(([False], active[:-1]))))


def derive_protocol_features(
    timeseries: pd.DataFrame,
    cells: pd.DataFrame | None = None,
    *,
    early_cycles: int = 30,
) -> pd.DataFrame:
    """Derive C-rate, voltage/SOC-window, and stage counts from observations."""

    required = {"cell_id", "cycle_index", "sample_index", "current_a", "voltage_v"}
    missing = required - set(timeseries.columns)
    if missing:
        raise ValueError(f"timeseries missing protocol feature columns: {sorted(missing)}")
    capacities: dict[str, float] = {}
    if cells is not None:
        if not {"cell_id", "nominal_capacity_ah"}.issubset(cells.columns):
            raise ValueError("cells requires cell_id and nominal_capacity_ah")
        capacities = {
            str(row["cell_id"]): float(row["nominal_capacity_ah"])
            for _, row in cells.iterrows()
        }
    cycle = pd.to_numeric(timeseries["cycle_index"], errors="coerce")
    bounded = timeseries.loc[cycle <= early_cycles].copy()
    bounded["cycle_index"] = cycle.loc[bounded.index]
    rows: list[dict[str, float | str]] = []
    for cell_id, group in bounded.groupby("cell_id", sort=True):
        group = group.sort_values(["cycle_index", "sample_index"])
        current = pd.to_numeric(group["current_a"], errors="coerce").to_numpy(dtype=float)
        voltage = pd.to_numeric(group["voltage_v"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(current) & np.isfinite(voltage)
        active = finite & (current > 1e-8)
        if "step_type" in group.columns:
            named_charge = group["step_type"].astype("string").str.lower().eq("charge").to_numpy()
            if np.sum(finite & named_charge) > 0:
                active = finite & named_charge
        charge_current = np.abs(current[active])
        charge_voltage = voltage[active]
        nominal = capacities.get(str(cell_id), float("nan"))
        stage_counts = []
        for _, cycle_group in group.groupby("cycle_index", sort=False):
            cycle_current = pd.to_numeric(cycle_group["current_a"], errors="coerce").to_numpy(dtype=float)
            stage_counts.append(_stage_count(np.isfinite(cycle_current) & (cycle_current > 1e-8)))
        row: dict[str, float | str] = {
            "cell_id": str(cell_id),
            "charge_current_a_median": float(np.median(charge_current)) if charge_current.size else float("nan"),
            "charge_current_a_peak": float(np.max(charge_current)) if charge_current.size else float("nan"),
            "charge_current_a_std": float(np.std(charge_current)) if charge_current.size else float("nan"),
            "charge_c_rate_median": float(np.median(charge_current) / nominal)
            if charge_current.size and np.isfinite(nominal) and nominal > 0
            else float("nan"),
            "charge_c_rate_peak": float(np.max(charge_current) / nominal)
            if charge_current.size and np.isfinite(nominal) and nominal > 0
            else float("nan"),
            "charge_voltage_v_min": float(np.min(charge_voltage)) if charge_voltage.size else float("nan"),
            "charge_voltage_v_max": float(np.max(charge_voltage)) if charge_voltage.size else float("nan"),
            "charge_stage_count_median": float(np.median(stage_counts)) if stage_counts else float("nan"),
        }
        if "charge_capacity_ah" in group.columns:
            q = pd.to_numeric(group.loc[active, "charge_capacity_ah"], errors="coerce").to_numpy(dtype=float)
            q = q[np.isfinite(q)]
            row["observed_charge_soc_window"] = (
                float(np.ptp(q) / nominal)
                if q.size and np.isfinite(nominal) and nominal > 0
                else float("nan")
            )
        else:
            row["observed_charge_soc_window"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
