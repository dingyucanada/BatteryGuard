"""Deterministic, offline Twin-0 charging surrogate.

Twin-0 is intentionally modest.  It is a transparent empirical model for
comparing charging policies, not a calibrated electrochemical model and never
a controller.  Its outputs are suitable for demos, regression tests, and the
deterministic :mod:`batteryguard.safety` envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from batteryguard.constants import SIMULATOR_VERSION
from batteryguard.schemas.policy import (
    ChargingPolicy,
    SimulationMetrics,
    SimulationTrajectory,
)
from batteryguard.simulator.interface import SimulationError


@dataclass(frozen=True, slots=True)
class Twin0Parameters:
    """Documented parameters for the fast empirical surrogate.

    The defaults roughly describe a small LFP/graphite research cell.  They are
    comparison parameters only; users must not interpret them as fitted cell
    properties.
    """

    nominal_capacity_ah: float = 1.1
    internal_resistance_ohm: float = 0.040
    coulombic_efficiency: float = 0.995
    thermal_time_constant_min: float = 18.0
    heat_gain_c_per_c2_min: float = 0.140
    time_step_min: float = 0.20
    max_steps: int = 25_000

    def __post_init__(self) -> None:
        positive = (
            self.nominal_capacity_ah,
            self.internal_resistance_ohm,
            self.coulombic_efficiency,
            self.thermal_time_constant_min,
            self.heat_gain_c_per_c2_min,
            self.time_step_min,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Twin-0 parameters must be finite and positive")
        if self.coulombic_efficiency > 1:
            raise ValueError("coulombic_efficiency cannot exceed 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


class Twin0Simulator:
    """Fast deterministic MSCC/CV policy simulator.

    No randomness, filesystem access, network access, or hardware API is used.
    Repeating a call with the same policy and conditions produces an identical
    Pydantic model (including the full trajectory).
    """

    def __init__(self, parameters: Twin0Parameters | None = None) -> None:
        self.parameters = parameters or Twin0Parameters()

    @property
    def version(self) -> str:
        return SIMULATOR_VERSION

    @staticmethod
    def _open_circuit_voltage(soc: float) -> float:
        """Smooth LFP-like OCV curve used solely by the empirical surrogate."""

        bounded_soc = min(max(soc, 0.0), 1.0)
        low_knee = 0.055 / (1.0 + math.exp(-(bounded_soc - 0.08) / 0.025))
        high_knee = 0.21 / (1.0 + math.exp(-(bounded_soc - 0.91) / 0.035))
        return 3.17 + low_knee + 0.13 * bounded_soc + high_knee

    @staticmethod
    def _plating_margin(soc: float, c_rate: float, temperature_c: float) -> float:
        """Return a transparent *risk proxy*, not a plating diagnosis."""

        high_soc_term = 0.10 * max(soc - 0.65, 0.0) / 0.35
        cold_term = 0.010 * max(20.0 - temperature_c, 0.0)
        return 0.25 - 0.055 * c_rate - high_soc_term - cold_term

    @staticmethod
    def _stage_rate(policy: ChargingPolicy, soc: float) -> float:
        for rate, upper_soc in zip(policy.c_rates, policy.soc_breaks, strict=True):
            if soc < upper_soc - 1e-12:
                return rate
        return policy.c_rates[-1]

    def simulate(
        self,
        policy: ChargingPolicy,
        *,
        initial_soc: float = 0.10,
        ambient_temperature_c: float = 25.0,
    ) -> SimulationTrajectory:
        """Simulate a policy and return samples plus comparison metrics.

        Expected numerical/runtime failures are encoded in a failed trajectory
        so the independent safety shield can deterministically choose FALLBACK.
        Programmer errors such as passing a non-policy object are also converted
        into a failure rather than escaping into policy optimization.
        """

        policy_id = getattr(policy, "policy_id", "<invalid-policy>")
        try:
            return self._simulate(policy, initial_soc, ambient_temperature_c)
        except Exception as exc:  # the safety boundary must see simulator failures
            error = f"Twin-0 simulation failed: {type(exc).__name__}: {exc}"
            return SimulationTrajectory(
                policy_id=str(policy_id),
                simulator_version=self.version,
                status="FAILED",
                time_min=[],
                voltage_v=[],
                current_c=[],
                temperature_c=[],
                soc=[],
                plating_margin=[],
                error=error,
            )

    def _simulate(
        self,
        policy: ChargingPolicy,
        initial_soc: float,
        ambient_temperature_c: float,
    ) -> SimulationTrajectory:
        if not isinstance(policy, ChargingPolicy):
            raise TypeError("policy must be a ChargingPolicy")
        if not math.isfinite(initial_soc) or not 0 <= initial_soc < policy.target_soc:
            raise ValueError("initial_soc must be finite and below target_soc")
        if not math.isfinite(ambient_temperature_c):
            raise ValueError("ambient_temperature_c must be finite")

        params = self.parameters
        time_min = [0.0]
        soc_trace = [float(initial_soc)]
        temperature_c = [float(ambient_temperature_c)]
        first_rate = self._stage_rate(policy, initial_soc)
        first_voltage = self._terminal_voltage(initial_soc, first_rate, policy)
        voltage_v = [first_voltage]
        current_c = [first_rate]
        plating_margin = [
            self._plating_margin(initial_soc, first_rate, ambient_temperature_c)
        ]

        degradation_proxy = 0.0
        energy_loss_wh = 0.0
        soc = float(initial_soc)
        temperature = float(ambient_temperature_c)

        for _ in range(params.max_steps):
            if soc >= policy.target_soc - 1e-10:
                break

            requested_rate = self._stage_rate(policy, soc)
            ocv = self._open_circuit_voltage(soc)
            resistance_drop_per_c = params.internal_resistance_ohm * params.nominal_capacity_ah
            voltage_limited_rate = max(
                policy.cv_cutoff_c,
                (policy.v_max - ocv) / max(resistance_drop_per_c, 1e-12),
            )
            applied_rate = min(requested_rate, voltage_limited_rate)
            if not math.isfinite(applied_rate) or applied_rate <= 0:
                raise SimulationError("non-positive current after voltage limiting")

            remaining_soc = policy.target_soc - soc
            max_dt_to_target = remaining_soc * 60.0 / (
                applied_rate * params.coulombic_efficiency
            )
            dt_min = min(params.time_step_min, max_dt_to_target)
            if dt_min <= 1e-12:
                soc = policy.target_soc
                break

            delta_soc = applied_rate * params.coulombic_efficiency * dt_min / 60.0
            next_soc = min(policy.target_soc, soc + delta_soc)

            heat_input = params.heat_gain_c_per_c2_min * applied_rate**2
            cooling = (temperature - ambient_temperature_c) / params.thermal_time_constant_min
            temperature += (heat_input - cooling) * dt_min

            current_a = applied_rate * params.nominal_capacity_ah
            energy_loss_wh += (
                current_a**2 * params.internal_resistance_ohm * dt_min / 60.0
            )
            thermal_factor = math.exp(max(temperature - 25.0, -20.0) / 28.0)
            high_soc_factor = 1.0 + 1.5 * max(next_soc - 0.75, 0.0)
            degradation_proxy += (
                applied_rate**1.65 * thermal_factor * high_soc_factor * dt_min / 60.0
            )

            soc = next_soc
            now = time_min[-1] + dt_min
            voltage = self._terminal_voltage(soc, applied_rate, policy)
            time_min.append(now)
            soc_trace.append(soc)
            temperature_c.append(temperature)
            voltage_v.append(voltage)
            current_c.append(applied_rate)
            plating_margin.append(self._plating_margin(soc, applied_rate, temperature))
        else:
            raise SimulationError("maximum simulation steps exceeded")

        if soc < policy.target_soc - 1e-7:
            raise SimulationError("target SOC was not reached")
        if len(time_min) < 2 or time_min[-1] <= 0:
            raise SimulationError("simulation produced no charging interval")

        metrics = SimulationMetrics(
            charge_time_min=time_min[-1],
            degradation_proxy=max(degradation_proxy, 0.0),
            max_temperature_c=max(temperature_c),
            energy_loss_wh=max(energy_loss_wh, 0.0),
            feasibility=all(
                (
                    math.isfinite(time_min[-1]),
                    math.isfinite(degradation_proxy),
                    math.isfinite(max(temperature_c)),
                    math.isfinite(energy_loss_wh),
                    abs(soc_trace[-1] - policy.target_soc) <= 1e-7,
                )
            ),
        )
        return SimulationTrajectory(
            policy_id=policy.policy_id,
            simulator_version=self.version,
            status="SUCCESS",
            time_min=time_min,
            voltage_v=voltage_v,
            current_c=current_c,
            temperature_c=temperature_c,
            soc=soc_trace,
            plating_margin=plating_margin,
            metrics=metrics,
        )

    def _terminal_voltage(
        self,
        soc: float,
        c_rate: float,
        policy: ChargingPolicy,
    ) -> float:
        current_a = c_rate * self.parameters.nominal_capacity_ah
        unconstrained = (
            self._open_circuit_voltage(soc)
            + current_a * self.parameters.internal_resistance_ohm
        )
        return min(unconstrained, policy.v_max)


# Short aliases retained for ergonomic imports and older examples.
Twin0 = Twin0Simulator
SurrogateSimulator = Twin0Simulator
