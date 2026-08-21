"""Generate deterministic, battery-shaped fixtures for the offline software demo.

The values are intentionally synthetic. They exercise data, model, uncertainty,
OOD, policy, safety, blind-reveal, and evidence paths without implying scientific
validation or redistributing a third-party dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoDataset:
    cells: pd.DataFrame
    cycles: pd.DataFrame
    timeseries: pd.DataFrame
    splits: pd.DataFrame
    dataset_id: str = "demo-synthetic-v1"


def _split_names(n_cells: int) -> list[str]:
    if n_cells < 30:
        raise ValueError("offline demo requires at least 30 cells")
    counts = {
        "train": n_cells - 28,
        "calibration": 10,
        "test": 8,
        "demo_hidden": 6,
        "external_ood": 4,
    }
    return [name for name, count in counts.items() for _ in range(count)]


def generate_demo_dataset(
    *, seed: int = 42, n_cells: int = 72, early_cycles: int = 30
) -> DemoDataset:
    """Return repeatable canonical tables with no network or file dependency."""

    if early_cycles < 10:
        raise ValueError("early_cycles must be at least 10")
    rng = np.random.default_rng(seed)
    split_names = _split_names(n_cells)
    rng.shuffle(split_names)
    hidden_indices = [index for index, split in enumerate(split_names) if split == "demo_hidden"]
    hidden_ood_index = hidden_indices[-2]

    cells: list[dict[str, object]] = []
    cycles: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    splits: list[dict[str, object]] = []

    for index in range(n_cells):
        cell_id = f"BG-{index + 1:04d}"
        split = split_names[index]
        external = split == "external_ood"
        hidden_ood = index == hidden_ood_index
        chemistry_ood = external or hidden_ood
        chemistry = "NMC_graphite" if chemistry_ood else "LFP_graphite"
        protocol_number = index % 6
        protocol_id = f"P-{protocol_number + 1}"
        base_c_rate = 1.0 + 0.38 * protocol_number
        latent_stress = float(np.clip(rng.normal(0.48, 0.18), 0.05, 0.95))
        manufacturing = float(rng.normal(0.0, 0.08))
        chemistry_penalty = 120.0 if chemistry_ood else 0.0
        cycle_life = int(
            np.clip(
                1780
                - 245 * base_c_rate
                - 610 * latent_stress
                + 190 * manufacturing
                - chemistry_penalty
                + rng.normal(0, 38),
                320,
                1650,
            )
        )
        nominal_capacity = float(1.10 + rng.normal(0, 0.012))
        ambient = 25.0 + (5.0 if external else 8.0 if hidden_ood else 0.0)

        cells.append(
            {
                "cell_id": cell_id,
                "chemistry": chemistry,
                "nominal_capacity_ah": nominal_capacity,
                "batch_id": f"B-{index // 12 + 1}",
                "protocol_id": protocol_id,
                "cycle_life": cycle_life,
                "eol_threshold": 0.80,
                "censored": False,
            }
        )
        splits.append(
            {
                "cell_id": cell_id,
                "split": split,
                "split_reason": "deterministic cell-level offline fixture",
                "group_keys": {"protocol_id": protocol_id, "batch_id": f"B-{index // 12 + 1}"},
                "ood_type": "chemistry_and_temperature" if chemistry_ood else None,
            }
        )

        degradation_rate = 0.20 / cycle_life
        for cycle_index in range(1, early_cycles + 1):
            normalized_capacity = 1.0 - degradation_rate * cycle_index
            early_knee = latent_stress * (cycle_index / early_cycles) ** 2 * 0.009
            noise = rng.normal(0, 0.0012)
            discharge_capacity = nominal_capacity * (
                normalized_capacity - early_knee + manufacturing * 0.008 + noise
            )
            charge_capacity = discharge_capacity / np.clip(
                0.996 - 0.004 * latent_stress + rng.normal(0, 0.0005), 0.975, 1.005
            )
            efficiency = discharge_capacity / charge_capacity
            dcir = 0.024 + 0.010 * latent_stress + cycle_index * (
                0.000018 + 0.000025 * latent_stress
            )
            max_temp = (
                ambient
                + 2.7 * base_c_rate**1.45
                + 3.2 * latent_stress
                + cycle_index * 0.012
                + rng.normal(0, 0.25)
            )
            charge_time = 3600 * 0.72 / base_c_rate + 390 + 65 * latent_stress
            cycles.append(
                {
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "charge_capacity_ah": charge_capacity,
                    "discharge_capacity_ah": discharge_capacity,
                    "coulombic_efficiency": efficiency,
                    "charge_time_s": charge_time + rng.normal(0, 9),
                    "discharge_time_s": 3600 + rng.normal(0, 15),
                    "dcir_ohm": dcir,
                    "avg_temp_c": ambient + 0.55 * (max_temp - ambient),
                    "max_temp_c": max_temp,
                    "charge_energy_wh": charge_capacity * 3.38,
                    "discharge_energy_wh": discharge_capacity * 3.24,
                }
            )

            if cycle_index not in {2, 10, early_cycles}:
                continue
            voltage_grid = np.linspace(2.0, 3.6, 65)
            curve_shift = degradation_rate * cycle_index * 2.8 + latent_stress * 0.002
            q_curve = discharge_capacity * (
                1.0
                / (1.0 + np.exp(-5.2 * (voltage_grid - 2.95 - curve_shift)))
            )
            for sample_index, (voltage, capacity) in enumerate(
                zip(voltage_grid, q_curve, strict=True)
            ):
                curves.append(
                    {
                        "cell_id": cell_id,
                        "cycle_index": cycle_index,
                        "sample_index": sample_index,
                        "time_s": float(sample_index * 55),
                        "current_a": -nominal_capacity,
                        "voltage_v": float(voltage),
                        "charge_capacity_ah": 0.0,
                        "discharge_capacity_ah": float(capacity),
                        "temperature_c": float(ambient + 1.0 + sample_index * 0.012),
                        "step_type": "discharge",
                    }
                )

    return DemoDataset(
        cells=pd.DataFrame(cells).sort_values("cell_id").reset_index(drop=True),
        cycles=pd.DataFrame(cycles).sort_values(["cell_id", "cycle_index"]).reset_index(
            drop=True
        ),
        timeseries=pd.DataFrame(curves)
        .sort_values(["cell_id", "cycle_index", "sample_index"])
        .reset_index(drop=True),
        splits=pd.DataFrame(splits).sort_values("cell_id").reset_index(drop=True),
    )
