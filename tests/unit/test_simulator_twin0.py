from __future__ import annotations

import os

from batteryguard.optimizer.policy_space import balanced_policy, fast_policy
from batteryguard.safety.shield import SafetyShield
from batteryguard.schemas.policy import SafetyDecision
from batteryguard.simulator.cache import SimulationCache, simulation_cache_key
from batteryguard.simulator.pybamm_backend import PyBaMMSimulator
from batteryguard.simulator.surrogate import Twin0Simulator


def test_simulator_twin0_is_deterministic_and_complete() -> None:
    simulator = Twin0Simulator()
    policy = balanced_policy()

    first = simulator.simulate(policy)
    second = simulator.simulate(policy)

    assert first == second
    assert first.status == "SUCCESS"
    assert first.metrics is not None and first.metrics.feasibility
    assert first.soc[-1] == policy.target_soc
    assert first.time_min[-1] == first.metrics.charge_time_min
    lengths = {
        len(first.time_min),
        len(first.voltage_v),
        len(first.current_c),
        len(first.temperature_c),
        len(first.soc),
        len(first.plating_margin),
    }
    assert lengths == {len(first.time_min)}
    assert len(first.time_min) > 2


def test_simulator_twin0_encodes_invalid_conditions_as_failure() -> None:
    result = Twin0Simulator().simulate(balanced_policy(), initial_soc=0.90)

    assert result.status == "FAILED"
    assert result.error is not None
    assert not result.time_min


def test_simulator_hot_environment_produces_repeatable_safety_reversal() -> None:
    simulator = Twin0Simulator()
    shield = SafetyShield()
    fast = fast_policy()
    balanced = balanced_policy()
    fast_trajectory = simulator.simulate(fast, ambient_temperature_c=40.0)
    balanced_trajectory = simulator.simulate(balanced, ambient_temperature_c=40.0)

    assert shield.evaluate(
        fast,
        fast_trajectory,
        ambient_temperature_c=40.0,
    ).decision == SafetyDecision.REJECT
    assert shield.evaluate(
        balanced,
        balanced_trajectory,
        ambient_temperature_c=40.0,
    ).decision == SafetyDecision.ALLOW


def test_simulator_pybamm_missing_dependency_degrades_clearly() -> None:
    simulator = PyBaMMSimulator(module_name="batteryguard_dependency_that_does_not_exist")

    result = simulator.simulate(balanced_policy())

    assert result.status == "SUCCESS"
    assert simulator.last_status.degraded
    assert simulator.last_status.mode == "twin0-fallback"
    assert "degraded-to-twin0" in result.simulator_version
    assert simulator.last_status.reason is not None


def test_simulator_pybamm_can_fail_closed_instead_of_degrading() -> None:
    simulator = PyBaMMSimulator(
        module_name="batteryguard_dependency_that_does_not_exist",
        allow_fallback=False,
    )

    result = simulator.simulate(balanced_policy())

    assert result.status == "FAILED"
    assert result.error and "not installed" in result.error


def test_simulator_pybamm_disables_optional_telemetry_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PYBAMM_DISABLE_TELEMETRY", raising=False)
    simulator = PyBaMMSimulator(module_name="math")

    module = simulator._load_module()

    assert module.__name__ == "math"
    assert os.environ["PYBAMM_DISABLE_TELEMETRY"] == "true"


def test_simulator_cache_key_covers_conditions_and_returns_copies() -> None:
    policy = balanced_policy()
    trajectory = Twin0Simulator().simulate(policy)
    key = simulation_cache_key(
        policy,
        simulator_version=trajectory.simulator_version,
        initial_soc=0.05,
        ambient_temperature_c=25.0,
    )
    warmer_key = simulation_cache_key(
        policy,
        simulator_version=trajectory.simulator_version,
        initial_soc=0.05,
        ambient_temperature_c=30.0,
    )
    cache = SimulationCache()
    cache.put(key, trajectory)

    cached = cache.get(key)
    assert cached == trajectory
    assert cached is not trajectory
    assert warmer_key != key
