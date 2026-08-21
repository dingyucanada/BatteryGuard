from __future__ import annotations

import math
from dataclasses import replace

import pytest

from batteryguard.optimizer.policy_space import balanced_policy, fast_policy
from batteryguard.safety.constraints import SafetyConstraints
from batteryguard.safety.shield import SafetyShield
from batteryguard.schemas.policy import SafetyDecision, SimulationTrajectory
from batteryguard.simulator.surrogate import Twin0Simulator


def _safe_pair() -> tuple[object, SimulationTrajectory]:
    policy = balanced_policy()
    trajectory = Twin0Simulator().simulate(policy)
    assert trajectory.status == "SUCCESS"
    return policy, trajectory


@pytest.mark.safety
def test_safety_known_good_policy_is_allowed() -> None:
    policy, trajectory = _safe_pair()

    result = SafetyShield().evaluate(policy, trajectory)

    assert result.decision == SafetyDecision.ALLOW
    assert not result.violations
    assert result.fallback_policy_id is None


@pytest.mark.safety
@pytest.mark.parametrize(
    ("field", "values", "expected_constraint"),
    [
        ("voltage_v", [3.7], "voltage"),
        ("current_c", [4.2], "current"),
        ("temperature_c", [46.0], "temperature"),
        ("soc", [0.95], "soc_window"),
        ("plating_margin", [-0.01], "plating_margin"),
    ],
)
def test_safety_every_hard_signal_violation_is_rejected(
    field: str,
    values: list[float],
    expected_constraint: str,
) -> None:
    policy, trajectory = _safe_pair()
    injected = [*getattr(trajectory, field)]
    injected[-1] = values[0]
    faulty = trajectory.model_copy(update={field: injected})

    result = SafetyShield().evaluate(policy, faulty)

    assert result.decision == SafetyDecision.REJECT
    assert expected_constraint in {violation.constraint for violation in result.violations}


@pytest.mark.safety
def test_safety_temperature_rise_and_rate_are_independently_checked() -> None:
    policy, trajectory = _safe_pair()
    length = len(trajectory.time_min)
    temperatures = [25.0 + 16.0 * index / (length - 1) for index in range(length)]
    rapid = trajectory.model_copy(update={"temperature_c": temperatures})
    tight = SafetyShield(
        SafetyConstraints(
            max_temperature_c=50.0,
            max_temperature_rise_c=10.0,
            max_temperature_rise_rate_c_per_min=0.2,
        )
    )

    result = tight.evaluate(policy, rapid, ambient_temperature_c=25.0)
    constraints = {violation.constraint for violation in result.violations}

    assert result.decision == SafetyDecision.REJECT
    assert "temperature_rise" in constraints
    assert "temperature_rise_rate" in constraints


@pytest.mark.safety
def test_safety_simulator_failure_triggers_fallback() -> None:
    policy = balanced_policy()
    failed = SimulationTrajectory(
        policy_id=policy.policy_id,
        simulator_version="fault-injection",
        status="FAILED",
        time_min=[],
        voltage_v=[],
        current_c=[],
        temperature_c=[],
        soc=[],
        plating_margin=[],
        error="injected convergence failure",
    )

    result = SafetyShield().evaluate(policy, failed)

    assert result.decision == SafetyDecision.FALLBACK
    assert result.fallback_policy_id == "conservative-cccv-v1"
    assert "simulator_status" in {item.constraint for item in result.violations}


@pytest.mark.safety
@pytest.mark.parametrize("fault", ["nan", "missing"])
def test_safety_nan_or_missing_signal_triggers_fallback(fault: str) -> None:
    policy, trajectory = _safe_pair()
    if fault == "nan":
        voltage = [*trajectory.voltage_v]
        voltage[1] = math.nan
    else:
        voltage = []
    faulty = trajectory.model_copy(update={"voltage_v": voltage})

    result = SafetyShield().evaluate(policy, faulty)

    assert result.decision == SafetyDecision.FALLBACK
    assert any(
        violation.constraint in {"non_finite_voltage", "missing_voltage"}
        for violation in result.violations
    )


@pytest.mark.safety
@pytest.mark.parametrize(("ood_score", "abstain"), [(0.9, False), (0.1, True)])
def test_safety_ood_or_abstain_blocks_aggressive_personalization(
    ood_score: float,
    abstain: bool,
) -> None:
    policy = fast_policy().model_copy(update={"personalized": True})
    trajectory = Twin0Simulator().simulate(policy)

    result = SafetyShield().evaluate(
        policy,
        trajectory,
        ood_score=ood_score,
        abstain=abstain,
    )

    assert result.decision == SafetyDecision.FALLBACK
    assert "personalization_restriction" in {
        violation.constraint for violation in result.violations
    }


@pytest.mark.safety
def test_safety_constraint_change_changes_case_hash() -> None:
    defaults = SafetyConstraints()
    tighter = replace(defaults, max_temperature_c=defaults.max_temperature_c - 1)

    assert defaults.safety_case_hash != tighter.safety_case_hash
    assert SafetyShield(defaults).safety_case_hash != SafetyShield(tighter).safety_case_hash


@pytest.mark.safety
def test_safety_learned_model_cannot_write_allow() -> None:
    policy, trajectory = _safe_pair()

    result = SafetyShield().evaluate(
        policy,
        trajectory,
        learned_decision=SafetyDecision.ALLOW,
    )

    assert result.decision == SafetyDecision.REJECT
    assert "decision_authority" in {item.constraint for item in result.violations}
