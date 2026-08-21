"""Optional PyBaMM Twin-1 adapter with an explicit Twin-0 degradation path."""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from batteryguard.schemas.policy import (
    ChargingPolicy,
    SimulationMetrics,
    SimulationTrajectory,
)
from batteryguard.simulator.interface import BackendUnavailableError, SimulationError
from batteryguard.simulator.surrogate import Twin0Simulator


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Observable status of the most recent optional-backend operation."""

    backend: str
    mode: str
    available: bool
    degraded: bool
    reason: str | None = None


class PyBaMMSimulator:
    """Lazy PyBaMM SPM adapter.

    PyBaMM is never imported at package import time.  If it is absent or a
    native solve fails, the default behavior is to return a deterministic
    Twin-0 trajectory whose ``simulator_version`` and :attr:`last_status`
    explicitly report the downgrade.  Set ``allow_fallback=False`` to receive a
    failed trajectory instead.
    """

    def __init__(
        self,
        *,
        allow_fallback: bool = True,
        fallback_simulator: Twin0Simulator | None = None,
        module_name: str = "pybamm",
        parameter_set: str = "Prada2013",
    ) -> None:
        self.allow_fallback = allow_fallback
        self.fallback_simulator = fallback_simulator or Twin0Simulator()
        self.module_name = module_name
        self.parameter_set = parameter_set
        available = self._module_available()
        reason = None if available else f"optional dependency '{module_name}' is not installed"
        self.last_status = BackendStatus(
            backend="pybamm",
            mode="ready" if available else "unavailable",
            available=available,
            degraded=False,
            reason=reason,
        )

    @property
    def version(self) -> str:
        return "pybamm-spm-adapter-v1"

    @property
    def available(self) -> bool:
        return self._module_available()

    def _module_available(self) -> bool:
        try:
            return importlib.util.find_spec(self.module_name) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def _load_module(self) -> ModuleType:
        if not self._module_available():
            raise BackendUnavailableError(
                f"optional dependency '{self.module_name}' is not installed"
            )
        # Optional PyBaMM releases may offer usage telemetry.  Keep the public
        # offline prototype quiet by default while preserving an explicit
        # operator override.
        os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
        return importlib.import_module(self.module_name)

    def simulate(
        self,
        policy: ChargingPolicy,
        *,
        initial_soc: float = 0.10,
        ambient_temperature_c: float = 25.0,
    ) -> SimulationTrajectory:
        try:
            pybamm = self._load_module()
            trajectory = self._simulate_native(
                pybamm,
                policy,
                initial_soc=initial_soc,
                ambient_temperature_c=ambient_temperature_c,
            )
            pybamm_version = str(getattr(pybamm, "__version__", "unknown"))
            self.last_status = BackendStatus(
                backend="pybamm",
                mode="native",
                available=True,
                degraded=False,
            )
            return trajectory.model_copy(
                update={"simulator_version": f"pybamm-{pybamm_version}-spm-v1"}
            )
        except Exception as exc:
            return self._handle_failure(
                policy,
                exc,
                initial_soc=initial_soc,
                ambient_temperature_c=ambient_temperature_c,
            )

    def _handle_failure(
        self,
        policy: ChargingPolicy,
        exc: Exception,
        *,
        initial_soc: float,
        ambient_temperature_c: float,
    ) -> SimulationTrajectory:
        reason = f"{type(exc).__name__}: {exc}"
        is_available = not isinstance(exc, BackendUnavailableError)
        if self.allow_fallback:
            fallback = self.fallback_simulator.simulate(
                policy,
                initial_soc=initial_soc,
                ambient_temperature_c=ambient_temperature_c,
            )
            self.last_status = BackendStatus(
                backend="pybamm",
                mode="twin0-fallback",
                available=is_available,
                degraded=True,
                reason=reason,
            )
            return fallback.model_copy(
                update={
                    "simulator_version": (
                        f"{self.version}:degraded-to-{self.fallback_simulator.version}"
                    )
                }
            )

        self.last_status = BackendStatus(
            backend="pybamm",
            mode="failed",
            available=is_available,
            degraded=False,
            reason=reason,
        )
        return SimulationTrajectory(
            policy_id=getattr(policy, "policy_id", "<invalid-policy>"),
            simulator_version=self.version,
            status="FAILED",
            time_min=[],
            voltage_v=[],
            current_c=[],
            temperature_c=[],
            soc=[],
            plating_margin=[],
            error=f"PyBaMM backend unavailable or failed: {reason}",
        )

    def _simulate_native(
        self,
        pybamm: ModuleType,
        policy: ChargingPolicy,
        *,
        initial_soc: float,
        ambient_temperature_c: float,
    ) -> SimulationTrajectory:
        """Run a small SPM experiment using only stable public PyBaMM APIs."""

        if not 0 <= initial_soc < policy.target_soc:
            raise ValueError("initial_soc must be below target_soc")
        if not math.isfinite(ambient_temperature_c):
            raise ValueError("ambient_temperature_c must be finite")

        commands: list[str] = []
        lower_soc = initial_soc
        for c_rate, upper_soc in zip(policy.c_rates, policy.soc_breaks, strict=True):
            stage_fraction = max(upper_soc - lower_soc, 0.0)
            duration_min = max(stage_fraction * 60.0 / c_rate, 0.01)
            commands.append(
                f"Charge at {c_rate} C for {duration_min} minutes or until {policy.v_max} V"
            )
            lower_soc = upper_soc

        model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})
        parameter_values = pybamm.ParameterValues(self.parameter_set)
        parameter_values.update(
            {
                "Ambient temperature [K]": ambient_temperature_c + 273.15,
                "Initial temperature [K]": ambient_temperature_c + 273.15,
            }
        )
        experiment = pybamm.Experiment(commands, period="12 seconds")
        simulation = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        solution = simulation.solve(initial_soc=initial_soc)

        time_seconds = self._entries(solution, "Time [s]")
        voltage_v = self._entries(solution, "Terminal voltage [V]")
        current_a = self._entries(solution, "Current [A]")
        temperature_k = self._entries(solution, "X-averaged cell temperature [K]")
        if not time_seconds:
            raise SimulationError("PyBaMM returned an empty solution")

        nominal_capacity = float(parameter_values["Nominal cell capacity [A.h]"])
        time_min = [value / 60.0 for value in time_seconds]
        current_c = [max(-value / nominal_capacity, 0.0) for value in current_a]
        temperature_c = [value - 273.15 for value in temperature_k]
        soc_trace = self._integrate_soc(time_seconds, current_c, initial_soc, policy.target_soc)
        plating_margin = [
            Twin0Simulator._plating_margin(soc, c_rate, temp)
            for soc, c_rate, temp in zip(
                soc_trace,
                current_c,
                temperature_c,
                strict=True,
            )
        ]

        degradation_proxy = 0.0
        energy_loss_wh = 0.0
        resistance = self.fallback_simulator.parameters.internal_resistance_ohm
        for index in range(1, len(time_min)):
            dt_min = max(time_min[index] - time_min[index - 1], 0.0)
            degradation_proxy += current_c[index] ** 1.65 * dt_min / 60.0
            current = current_c[index] * nominal_capacity
            energy_loss_wh += current**2 * resistance * dt_min / 60.0

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
            metrics=SimulationMetrics(
                charge_time_min=max(time_min[-1] - time_min[0], 1e-12),
                degradation_proxy=max(degradation_proxy, 0.0),
                max_temperature_c=max(temperature_c),
                energy_loss_wh=max(energy_loss_wh, 0.0),
                feasibility=soc_trace[-1] >= policy.target_soc - 0.02,
            ),
        )

    @staticmethod
    def _entries(solution: Any, variable: str) -> list[float]:
        values = solution[variable].entries
        flattened = values.reshape(-1) if hasattr(values, "reshape") else values
        return [float(value) for value in flattened]

    @staticmethod
    def _integrate_soc(
        time_seconds: list[float],
        current_c: list[float],
        initial_soc: float,
        target_soc: float,
    ) -> list[float]:
        soc = initial_soc
        output = [soc]
        for index in range(1, len(time_seconds)):
            dt_hours = max(time_seconds[index] - time_seconds[index - 1], 0.0) / 3600.0
            rate = 0.5 * (current_c[index] + current_c[index - 1])
            soc = min(target_soc, soc + rate * dt_hours)
            output.append(soc)
        return output


PyBaMMBackend = PyBaMMSimulator
