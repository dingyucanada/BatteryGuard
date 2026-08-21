"""Safety-gated deterministic Pareto selection."""

from __future__ import annotations

from collections.abc import Iterable

from batteryguard.constants import SAFETY_VERSION
from batteryguard.optimizer.objectives import dominates, objective_vector
from batteryguard.schemas.policy import EvaluatedPolicy, SafetyDecision


def is_safety_feasible(
    candidate: EvaluatedPolicy,
    *,
    expected_safety_case_hash: str | None = None,
) -> bool:
    """Require an explicit shield ALLOW plus a successful feasible simulation."""

    trajectory = candidate.trajectory
    if candidate.safety.decision != SafetyDecision.ALLOW:
        return False
    if candidate.safety.violations or candidate.safety.fallback_policy_id is not None:
        return False
    if candidate.safety.policy_id != candidate.policy.policy_id:
        return False
    if trajectory.policy_id != candidate.policy.policy_id:
        return False
    if candidate.safety.shield_version != SAFETY_VERSION:
        return False
    safety_hash = candidate.safety.safety_case_hash
    if len(safety_hash) != 64 or any(character not in "0123456789abcdef" for character in safety_hash):
        return False
    if expected_safety_case_hash is not None and safety_hash != expected_safety_case_hash:
        return False
    if trajectory.status != "SUCCESS" or trajectory.metrics is None:
        return False
    if not trajectory.metrics.feasibility:
        return False
    try:
        return objective_vector(candidate).finite
    except (TypeError, ValueError):
        return False


def pareto_front(
    candidates: Iterable[EvaluatedPolicy],
    *,
    expected_safety_case_hash: str | None = None,
) -> list[EvaluatedPolicy]:
    """Return only safe, feasible, non-dominated candidates in input order."""

    feasible = [
        candidate
        for candidate in candidates
        if is_safety_feasible(
            candidate,
            expected_safety_case_hash=expected_safety_case_hash,
        )
    ]
    front: list[EvaluatedPolicy] = []
    for candidate in feasible:
        candidate_objectives = objective_vector(candidate)
        if any(
            other is not candidate
            and dominates(objective_vector(other), candidate_objectives)
            for other in feasible
        ):
            continue
        front.append(candidate)
    return front


def pareto_policy_ids(
    candidates: Iterable[EvaluatedPolicy],
    *,
    expected_safety_case_hash: str | None = None,
) -> list[str]:
    return [
        candidate.policy.policy_id
        for candidate in pareto_front(
            candidates,
            expected_safety_case_hash=expected_safety_case_hash,
        )
    ]


def mark_pareto_optimal(
    candidates: Iterable[EvaluatedPolicy],
    *,
    expected_safety_case_hash: str | None = None,
) -> list[EvaluatedPolicy]:
    """Return copies with truthful ``pareto_optimal`` flags for every candidate."""

    materialized = list(candidates)
    front_ids = {
        id(candidate)
        for candidate in pareto_front(
            materialized,
            expected_safety_case_hash=expected_safety_case_hash,
        )
    }
    return [
        candidate.model_copy(update={"pareto_optimal": id(candidate) in front_ids})
        for candidate in materialized
    ]


select_pareto = pareto_front
compute_pareto_front = pareto_front
