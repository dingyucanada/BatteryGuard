"""Orchestration for simulate -> shield -> Pareto policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from batteryguard.optimizer.pareto import mark_pareto_optimal, pareto_policy_ids
from batteryguard.optimizer.policy_space import default_policies
from batteryguard.safety.shield import SafetyShield
from batteryguard.schemas.policy import (
    ChargingPolicy,
    EvaluatedPolicy,
    PolicyResponse,
    SafetyDecision,
    SimulationTrajectory,
)
from batteryguard.simulator.interface import BatterySimulator


@dataclass(frozen=True, slots=True)
class SearchContext:
    initial_soc: float = 0.10
    ambient_temperature_c: float = 25.0
    ood_score: float = 0.0
    abstain: bool = False


class PolicySearch:
    """Evaluate transparent candidates while preserving SafetyShield authority."""

    def __init__(self, simulator: BatterySimulator, shield: SafetyShield) -> None:
        self.simulator = simulator
        self.shield = shield

    def evaluate_policy(
        self,
        policy: ChargingPolicy,
        *,
        context: SearchContext | None = None,
    ) -> EvaluatedPolicy:
        resolved = context or SearchContext()
        try:
            trajectory = self.simulator.simulate(
                policy,
                initial_soc=resolved.initial_soc,
                ambient_temperature_c=resolved.ambient_temperature_c,
            )
        except Exception as exc:
            trajectory = SimulationTrajectory(
                policy_id=policy.policy_id,
                simulator_version=getattr(self.simulator, "version", "unknown"),
                status="FAILED",
                time_min=[],
                voltage_v=[],
                current_c=[],
                temperature_c=[],
                soc=[],
                plating_margin=[],
                error=f"Simulator exception: {type(exc).__name__}: {exc}",
            )
        safety = self.shield.evaluate(
            policy,
            trajectory,
            ambient_temperature_c=resolved.ambient_temperature_c,
            ood_score=resolved.ood_score,
            abstain=resolved.abstain,
        )
        return EvaluatedPolicy(policy=policy, trajectory=trajectory, safety=safety)

    def evaluate(
        self,
        policies: Iterable[ChargingPolicy] | None = None,
        *,
        context: SearchContext | None = None,
    ) -> list[EvaluatedPolicy]:
        candidates = list(policies) if policies is not None else default_policies()
        evaluated = [
            self.evaluate_policy(policy, context=context)
            for policy in candidates
        ]
        return mark_pareto_optimal(
            evaluated,
            expected_safety_case_hash=self.shield.safety_case_hash,
        )

    def response(
        self,
        cell_id: str,
        policies: Iterable[ChargingPolicy] | None = None,
        *,
        context: SearchContext | None = None,
        evidence_ids: Iterable[str] = (),
    ) -> PolicyResponse:
        evaluated = self.evaluate(policies, context=context)
        rejected = [
            candidate.policy.policy_id
            for candidate in evaluated
            if candidate.safety.decision == SafetyDecision.REJECT
        ]
        needs_fallback = any(
            candidate.safety.decision != SafetyDecision.ALLOW for candidate in evaluated
        )
        return PolicyResponse(
            cell_id=cell_id,
            policies=evaluated,
            pareto_front=pareto_policy_ids(
                evaluated,
                expected_safety_case_hash=self.shield.safety_case_hash,
            ),
            rejected=rejected,
            fallback=self.shield.fallback_policy_id if needs_fallback else None,
            evidence_ids=list(evidence_ids),
        )


PolicyOptimizer = PolicySearch


def evaluate_candidates(
    policies: Iterable[ChargingPolicy],
    *,
    simulator: BatterySimulator,
    shield: SafetyShield,
    context: SearchContext | None = None,
) -> list[EvaluatedPolicy]:
    return PolicySearch(simulator, shield).evaluate(policies, context=context)
