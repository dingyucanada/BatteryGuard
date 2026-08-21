from __future__ import annotations

from batteryguard.optimizer.objectives import ObjectiveVector, dominates
from batteryguard.optimizer.pareto import mark_pareto_optimal, pareto_front
from batteryguard.optimizer.policy_space import default_policies, get_policy_library
from batteryguard.optimizer.search import PolicySearch, SearchContext
from batteryguard.safety.shield import SafetyShield
from batteryguard.schemas.policy import (
    EvaluatedPolicy,
    PolicyFamily,
    SafetyDecision,
    SafetyResult,
    SimulationMetrics,
)
from batteryguard.simulator.surrogate import Twin0Simulator


def test_optimizer_library_contains_all_required_strategy_families() -> None:
    library = get_policy_library()

    assert set(library) == set(PolicyFamily)
    assert [policy.family for policy in default_policies(include_fallback=True)] == [
        PolicyFamily.FAST,
        PolicyFamily.BALANCED,
        PolicyFamily.LIFE,
        PolicyFamily.FALLBACK,
    ]
    assert library[PolicyFamily.FAST].soc_breaks == [0.25, 0.5, 0.72, 0.8]
    assert all(not policy.personalized for policy in library.values())


def test_optimizer_dominance_uses_all_minimization_objectives() -> None:
    better = ObjectiveVector(20.0, 1.0, 30.0, 0.1)
    worse = ObjectiveVector(25.0, 1.2, 31.0, 0.2)
    tradeoff = ObjectiveVector(15.0, 2.0, 32.0, 0.3)

    assert dominates(better, worse)
    assert not dominates(worse, better)
    assert not dominates(better, tradeoff)


def test_optimizer_pareto_excludes_every_non_allowed_candidate() -> None:
    simulator = Twin0Simulator()
    shield = SafetyShield()
    safe = PolicySearch(simulator, shield).evaluate_policy(default_policies()[1])
    assert safe.safety.decision == SafetyDecision.ALLOW

    unsafe_policy = safe.policy.model_copy(
        update={"policy_id": "unsafe", "v_max": 4.2}
    )
    unsafe_trajectory = simulator.simulate(unsafe_policy)
    unsafe = EvaluatedPolicy(
        policy=unsafe_policy,
        trajectory=unsafe_trajectory,
        safety=shield.evaluate(unsafe_policy, unsafe_trajectory),
    )
    assert unsafe.safety.decision == SafetyDecision.REJECT

    front = pareto_front([unsafe, safe])
    assert [candidate.policy.policy_id for candidate in front] == [safe.policy.policy_id]


def test_optimizer_pareto_marks_dominated_allowed_policy_false() -> None:
    search = PolicySearch(Twin0Simulator(), SafetyShield())
    base = search.evaluate_policy(default_policies()[1])
    assert base.trajectory.metrics is not None
    dominated_metrics = SimulationMetrics(
        charge_time_min=base.trajectory.metrics.charge_time_min + 10,
        degradation_proxy=base.trajectory.metrics.degradation_proxy + 1,
        max_temperature_c=base.trajectory.metrics.max_temperature_c + 1,
        energy_loss_wh=base.trajectory.metrics.energy_loss_wh + 1,
        feasibility=True,
    )
    dominated = base.model_copy(
        update={
            "policy": base.policy.model_copy(update={"policy_id": "dominated"}),
            "trajectory": base.trajectory.model_copy(
                update={"policy_id": "dominated", "metrics": dominated_metrics}
            ),
            "safety": base.safety.model_copy(update={"policy_id": "dominated"}),
        }
    )

    marked = mark_pareto_optimal([base, dominated])
    assert marked[0].pareto_optimal
    assert not marked[1].pareto_optimal


def test_optimizer_pareto_rejects_unverified_learned_allow() -> None:
    candidate = PolicySearch(Twin0Simulator(), SafetyShield()).evaluate_policy(
        default_policies()[1]
    )
    forged = candidate.model_copy(
        update={
            "safety": SafetyResult(
                decision=SafetyDecision.ALLOW,
                policy_id=candidate.policy.policy_id,
                shield_version="learned-model-v1",
                safety_case_hash="model-said-allow",
            )
        }
    )

    assert pareto_front([forged]) == []


def test_optimizer_simulator_exception_cannot_bypass_shield() -> None:
    class ExplodingSimulator:
        version = "exploding-test-v1"

        def simulate(self, policy: object, **_: object) -> object:
            raise RuntimeError("injected failure")

    evaluated = PolicySearch(ExplodingSimulator(), SafetyShield()).evaluate_policy(
        default_policies()[0]
    )

    assert evaluated.trajectory.status == "FAILED"
    assert evaluated.safety.decision == SafetyDecision.FALLBACK
    assert not evaluated.pareto_optimal


def test_optimizer_hot_search_rejects_fast_and_exposes_fallback() -> None:
    search = PolicySearch(Twin0Simulator(), SafetyShield())

    response = search.response(
        "hot-demo",
        context=SearchContext(ambient_temperature_c=40.0),
    )

    assert "fast-mscc-v1" in response.rejected
    assert "fast-mscc-v1" not in response.pareto_front
    assert response.fallback == "conservative-cccv-v1"
