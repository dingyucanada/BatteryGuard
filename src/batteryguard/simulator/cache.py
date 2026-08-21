"""Small deterministic in-memory cache for simulation results."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from threading import RLock
from typing import Any

from batteryguard.schemas.policy import ChargingPolicy, SimulationTrajectory


def simulation_cache_key(
    policy: ChargingPolicy,
    *,
    simulator_version: str,
    initial_soc: float,
    ambient_temperature_c: float,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable content key for all simulation-affecting inputs."""

    numeric = (initial_soc, ambient_temperature_c)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("cache-key conditions must be finite")
    payload = {
        "policy": policy.model_dump(mode="json"),
        "simulator_version": simulator_version,
        "initial_soc": initial_soc,
        "ambient_temperature_c": ambient_temperature_c,
        "extra": dict(extra or {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SimulationCache:
    """Thread-safe process-local cache that returns defensive model copies."""

    def __init__(self) -> None:
        self._values: dict[str, SimulationTrajectory] = {}
        self._lock = RLock()

    def get(self, key: str) -> SimulationTrajectory | None:
        with self._lock:
            value = self._values.get(key)
            return None if value is None else value.model_copy(deep=True)

    def put(self, key: str, value: SimulationTrajectory) -> None:
        with self._lock:
            self._values[key] = value.model_copy(deep=True)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


TrajectoryCache = SimulationCache
