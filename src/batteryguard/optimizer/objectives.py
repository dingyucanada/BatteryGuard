"""Objective extraction and dominance helpers for charging policies."""

from __future__ import annotations

import math
from dataclasses import dataclass

from batteryguard.schemas.policy import EvaluatedPolicy, SimulationMetrics, SimulationTrajectory


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    """All four MVP minimization objectives in a stable order."""

    charge_time_min: float
    degradation_proxy: float
    max_temperature_c: float
    energy_loss_wh: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.charge_time_min,
            self.degradation_proxy,
            self.max_temperature_c,
            self.energy_loss_wh,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "charge_time_min": self.charge_time_min,
            "degradation_proxy": self.degradation_proxy,
            "max_temperature_c": self.max_temperature_c,
            "energy_loss_wh": self.energy_loss_wh,
        }

    @property
    def finite(self) -> bool:
        return all(math.isfinite(value) for value in self.as_tuple())


def objective_vector(
    value: EvaluatedPolicy | SimulationTrajectory | SimulationMetrics,
) -> ObjectiveVector:
    """Extract objective values, raising on failed/incomplete simulations."""

    if isinstance(value, EvaluatedPolicy):
        metrics = value.trajectory.metrics
    elif isinstance(value, SimulationTrajectory):
        metrics = value.metrics
    elif isinstance(value, SimulationMetrics):
        metrics = value
    else:
        raise TypeError("expected EvaluatedPolicy, SimulationTrajectory, or SimulationMetrics")
    if metrics is None:
        raise ValueError("simulation metrics are required for objective comparison")
    vector = ObjectiveVector(
        charge_time_min=metrics.charge_time_min,
        degradation_proxy=metrics.degradation_proxy,
        max_temperature_c=metrics.max_temperature_c,
        energy_loss_wh=metrics.energy_loss_wh,
    )
    if not vector.finite:
        raise ValueError("objective vector contains non-finite values")
    return vector


def dominates(left: ObjectiveVector, right: ObjectiveVector, *, tolerance: float = 1e-12) -> bool:
    """Return true when ``left`` weakly improves all and strictly improves one."""

    left_values = left.as_tuple()
    right_values = right.as_tuple()
    no_worse = all(a <= b + tolerance for a, b in zip(left_values, right_values, strict=True))
    strictly_better = any(
        a < b - tolerance for a, b in zip(left_values, right_values, strict=True)
    )
    return no_worse and strictly_better


extract_objectives = objective_vector
