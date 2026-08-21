"""Common contracts for offline charging simulators.

Simulators in this package are deliberately one-way research components: they
accept an in-memory policy and return an in-memory trajectory.  They contain no
hardware, network, or device-control integration.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from batteryguard.schemas.policy import ChargingPolicy, SimulationTrajectory


class SimulationError(RuntimeError):
    """Raised internally when a simulator cannot produce a valid trajectory."""


class BackendUnavailableError(SimulationError):
    """Raised when an optional simulator backend is not installed."""


@runtime_checkable
class BatterySimulator(Protocol):
    """Minimal simulator interface used by policy search and the API layer."""

    @property
    def version(self) -> str:
        """Return a stable implementation/version identifier."""

    def simulate(
        self,
        policy: ChargingPolicy,
        *,
        initial_soc: float = 0.10,
        ambient_temperature_c: float = 25.0,
    ) -> SimulationTrajectory:
        """Evaluate ``policy`` without communicating with external hardware."""


# Backwards-friendly names for callers that prefer a generic interface name.
Simulator = BatterySimulator
SimulatorBackend = BatterySimulator
