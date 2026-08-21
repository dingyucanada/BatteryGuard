"""Deterministic, model-independent charging-policy safety shield."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from batteryguard.constants import SAFETY_VERSION
from batteryguard.safety.constraints import SafetyConstraints
from batteryguard.safety.fallback import FALLBACK_POLICY_ID
from batteryguard.schemas.policy import (
    ChargingPolicy,
    ConstraintViolation,
    PolicyFamily,
    SafetyDecision,
    SafetyResult,
    SimulationTrajectory,
)
from batteryguard.schemas.prediction import PredictionResponse


@dataclass(frozen=True, slots=True)
class _Finding:
    violation: ConstraintViolation
    hard: bool


class SafetyShield:
    """Apply immutable safety rules without consulting learned model outputs.

    A predictor may provide OOD/abstention context and may propose a policy, but
    only this shield computes the final :class:`SafetyDecision`.  Calls are
    deterministic and contain no I/O.
    """

    def __init__(
        self,
        constraints: SafetyConstraints | Mapping[str, Any] | Any | None = None,
        *,
        fallback_policy_id: str | None = None,
    ) -> None:
        configured_fallback: Any = None
        if constraints is None:
            self.constraints = SafetyConstraints()
        elif isinstance(constraints, SafetyConstraints):
            self.constraints = constraints
        else:
            if isinstance(constraints, Mapping):
                raw_constraints = constraints
            else:
                model_dump = getattr(constraints, "model_dump", None)
                if not callable(model_dump):
                    raise TypeError("constraints must be SafetyConstraints, settings, or a mapping")
                raw_constraints = model_dump()
            configured_fallback = raw_constraints.get("fallback_policy_id")
            self.constraints = SafetyConstraints.from_mapping(raw_constraints)
        resolved_fallback = fallback_policy_id or configured_fallback or FALLBACK_POLICY_ID
        if not isinstance(resolved_fallback, str) or not resolved_fallback:
            raise ValueError("fallback_policy_id cannot be empty")
        self.fallback_policy_id = resolved_fallback

    @property
    def version(self) -> str:
        return SAFETY_VERSION

    @property
    def safety_case_hash(self) -> str:
        return self.constraints.safety_case_hash

    def evaluate(
        self,
        policy: ChargingPolicy,
        trajectory: SimulationTrajectory | None,
        *,
        ambient_temperature_c: float | None = None,
        ood_score: float | None = None,
        ood: bool | None = None,
        abstain: bool | None = None,
        prediction: PredictionResponse | None = None,
        learned_decision: SafetyDecision | str | None = None,
    ) -> SafetyResult:
        """Return ALLOW, REJECT, or FALLBACK from deterministic checks.

        ``learned_decision`` exists only as a guarded integration seam.  If an
        upstream learned component attempts to supply any final decision, the
        attempt is recorded and the shield refuses ALLOW.
        """

        policy_id = str(getattr(policy, "policy_id", "<unknown-policy>"))
        try:
            return self._evaluate(
                policy,
                trajectory,
                ambient_temperature_c=ambient_temperature_c,
                ood_score=ood_score,
                ood=ood,
                abstain=abstain,
                prediction=prediction,
                learned_decision=learned_decision,
            )
        except Exception as exc:
            violation = ConstraintViolation(
                constraint="shield_exception",
                value=type(exc).__name__,
                limit="successful deterministic evaluation",
                message=f"Safety evaluation failed closed: {exc}",
            )
            return self._result(
                SafetyDecision.FALLBACK,
                policy_id,
                [violation],
            )

    # Common naming variants used by orchestration code.
    check = evaluate
    assess = evaluate

    def _evaluate(
        self,
        policy: ChargingPolicy,
        trajectory: SimulationTrajectory | None,
        *,
        ambient_temperature_c: float | None,
        ood_score: float | None,
        ood: bool | None,
        abstain: bool | None,
        prediction: PredictionResponse | None,
        learned_decision: SafetyDecision | str | None,
    ) -> SafetyResult:
        if not isinstance(policy, ChargingPolicy):
            raise TypeError("policy must be a ChargingPolicy")

        findings: list[_Finding] = []
        constraints = self.constraints

        if learned_decision is not None:
            findings.append(
                self._finding(
                    "decision_authority",
                    str(learned_decision),
                    "SafetyShield only",
                    "A learned component attempted to write the final safety decision",
                    hard=True,
                )
            )

        self._check_policy(policy, findings)

        resolved_ood_score = ood_score
        personalization_context_missing = (
            prediction is None and ood_score is None and ood is None and abstain is None
        )
        if abstain is not None and not isinstance(abstain, bool):
            findings.append(
                self._finding(
                    "abstention_signal",
                    str(abstain),
                    "boolean",
                    "Abstention signal has the wrong type",
                )
            )
        resolved_abstain = bool(abstain) if abstain is not None else False
        if prediction is not None:
            resolved_ood_score = prediction.ood_score
            resolved_abstain = prediction.abstain
        if resolved_ood_score is None:
            resolved_ood_score = 0.0
        if not isinstance(resolved_ood_score, (int, float)) or not math.isfinite(
            float(resolved_ood_score)
        ):
            findings.append(
                self._finding(
                    "ood_signal",
                    resolved_ood_score,
                    "finite score in [0, 1]",
                    "OOD score is missing or non-finite",
                )
            )
            resolved_ood_score = 1.0
        elif not 0 <= float(resolved_ood_score) <= 1:
            findings.append(
                self._finding(
                    "ood_signal",
                    float(resolved_ood_score),
                    "[0, 1]",
                    "OOD score lies outside its contract",
                )
            )

        aggressive = (
            policy.family == PolicyFamily.FAST
            or max(policy.c_rates, default=0.0) > constraints.aggressive_c_rate
        )
        if policy.personalized and aggressive and personalization_context_missing:
            findings.append(
                self._finding(
                    "personalization_evidence",
                    None,
                    "explicit OOD and abstention context",
                    "Aggressive personalization lacks uncertainty evidence",
                )
            )
        ood_condition = bool(ood) or float(resolved_ood_score) > constraints.ood_threshold
        if policy.personalized and aggressive and (ood_condition or resolved_abstain):
            reasons: list[str] = []
            if ood_condition:
                reasons.append("OOD")
            if resolved_abstain:
                reasons.append("abstention")
            findings.append(
                self._finding(
                    "personalization_restriction",
                    "+".join(reasons),
                    "in-distribution non-abstained evidence",
                    "Aggressive personalized charging is disabled for uncertain evidence",
                )
            )

        if trajectory is None:
            findings.append(
                self._finding(
                    "missing_trajectory",
                    None,
                    "complete successful trajectory",
                    "No simulator trajectory was supplied",
                )
            )
            return self._decide(policy.policy_id, findings)

        if not isinstance(trajectory, SimulationTrajectory):
            findings.append(
                self._finding(
                    "invalid_trajectory",
                    type(trajectory).__name__,
                    "SimulationTrajectory",
                    "Simulator output has the wrong type",
                )
            )
            return self._decide(policy.policy_id, findings)

        if trajectory.policy_id != policy.policy_id:
            findings.append(
                self._finding(
                    "policy_integrity",
                    trajectory.policy_id,
                    policy.policy_id,
                    "Trajectory policy ID does not match the evaluated policy",
                )
            )
        if trajectory.status != "SUCCESS":
            findings.append(
                self._finding(
                    "simulator_status",
                    trajectory.status,
                    "SUCCESS",
                    trajectory.error or "Simulator did not converge",
                )
            )
            return self._decide(policy.policy_id, findings)
        if trajectory.metrics is None:
            findings.append(
                self._finding(
                    "simulation_metrics",
                    None,
                    "complete metrics",
                    "Successful trajectory is missing simulation metrics",
                )
            )
        elif not trajectory.metrics.feasibility:
            findings.append(
                self._finding(
                    "simulator_feasibility",
                    False,
                    True,
                    "Simulator marked the trajectory infeasible",
                )
            )
        elif not all(
            math.isfinite(value)
            for value in (
                trajectory.metrics.charge_time_min,
                trajectory.metrics.degradation_proxy,
                trajectory.metrics.max_temperature_c,
                trajectory.metrics.energy_loss_wh,
            )
        ):
            findings.append(
                self._finding(
                    "non_finite_metrics",
                    "NaN or infinity",
                    "finite simulation metrics",
                    "Simulation metrics contain a non-finite value",
                )
            )

        arrays = {
            "time": trajectory.time_min,
            "voltage": trajectory.voltage_v,
            "current": trajectory.current_c,
            "temperature": trajectory.temperature_c,
            "soc": trajectory.soc,
            "plating_margin": trajectory.plating_margin,
        }
        expected_length = len(trajectory.time_min)
        valid_arrays: dict[str, list[float]] = {}
        for name, values in arrays.items():
            converted = self._validated_array(name, values, expected_length, findings)
            if converted is not None:
                valid_arrays[name] = converted

        times = valid_arrays.get("time")
        if times is not None:
            if len(times) < 2:
                findings.append(
                    self._finding(
                        "trajectory_length",
                        len(times),
                        ">= 2",
                        "Trajectory has no evaluable charging interval",
                    )
                )
            elif any(
                next_time <= time
                for time, next_time in zip(times, times[1:], strict=False)
            ):
                findings.append(
                    self._finding(
                        "time_monotonicity",
                        "non-increasing",
                        "strictly increasing",
                        "Trajectory time must increase at every sample",
                    )
                )

        self._check_numeric_limits(valid_arrays, ambient_temperature_c, findings)

        soc_values = valid_arrays.get("soc")
        if soc_values and soc_values[-1] < policy.target_soc - 1e-5:
            findings.append(
                self._finding(
                    "target_soc_reached",
                    soc_values[-1],
                    policy.target_soc,
                    "Simulator stopped before reaching the policy target SOC",
                )
            )

        return self._decide(policy.policy_id, findings)

    def _check_policy(self, policy: ChargingPolicy, findings: list[_Finding]) -> None:
        constraints = self.constraints
        numeric_policy_values = [
            policy.v_max,
            policy.cv_cutoff_c,
            policy.target_soc,
            *policy.c_rates,
            *policy.soc_breaks,
        ]
        if not all(math.isfinite(value) for value in numeric_policy_values):
            findings.append(
                self._finding(
                    "non_finite_policy",
                    "NaN or infinity",
                    "finite policy parameters",
                    "Policy contains a non-finite parameter",
                )
            )
        if policy.v_max > constraints.max_voltage_v:
            findings.append(
                self._finding(
                    "policy_voltage",
                    policy.v_max,
                    constraints.max_voltage_v,
                    "Requested policy voltage exceeds the hard maximum",
                    hard=True,
                )
            )
        max_rate = max([*policy.c_rates, policy.cv_cutoff_c], default=math.inf)
        if max_rate > constraints.max_c_rate:
            findings.append(
                self._finding(
                    "policy_current",
                    max_rate,
                    constraints.max_c_rate,
                    "Requested policy C-rate exceeds the hard maximum",
                    hard=True,
                )
            )
        breaks = [*policy.soc_breaks, policy.target_soc]
        outside = [
            value
            for value in breaks
            if value < constraints.min_soc or value > constraints.max_soc
        ]
        if outside:
            findings.append(
                self._finding(
                    "policy_soc_window",
                    max(outside),
                    f"[{constraints.min_soc}, {constraints.max_soc}]",
                    "Requested policy SOC lies outside the hard window",
                    hard=True,
                )
            )

    def _validated_array(
        self,
        name: str,
        values: Sequence[Any] | None,
        expected_length: int,
        findings: list[_Finding],
    ) -> list[float] | None:
        if values is None or len(values) == 0:
            findings.append(
                self._finding(
                    f"missing_{name}",
                    None,
                    "non-empty finite signal",
                    f"Trajectory is missing the {name} safety signal",
                )
            )
            return None
        if len(values) != expected_length:
            findings.append(
                self._finding(
                    f"{name}_length",
                    len(values),
                    expected_length,
                    f"{name} signal length differs from trajectory time",
                )
            )
            return None

        converted: list[float] = []
        for value in values:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                findings.append(
                    self._finding(
                        f"non_finite_{name}",
                        str(value),
                        "finite numeric signal",
                        f"{name} contains NaN, infinity, or a non-numeric value",
                    )
                )
                return None
            converted.append(float(value))
        return converted

    def _check_numeric_limits(
        self,
        arrays: dict[str, list[float]],
        ambient_temperature_c: float | None,
        findings: list[_Finding],
    ) -> None:
        constraints = self.constraints

        voltages = arrays.get("voltage")
        if voltages:
            if max(voltages) > constraints.max_voltage_v:
                findings.append(
                    self._finding(
                        "voltage",
                        max(voltages),
                        constraints.max_voltage_v,
                        "Trajectory exceeds the maximum voltage",
                        hard=True,
                    )
                )
            if min(voltages) < constraints.min_voltage_v:
                findings.append(
                    self._finding(
                        "voltage_minimum",
                        min(voltages),
                        constraints.min_voltage_v,
                        "Trajectory falls below the minimum voltage",
                        hard=True,
                    )
                )

        currents = arrays.get("current")
        if currents:
            if max(abs(value) for value in currents) > constraints.max_c_rate:
                findings.append(
                    self._finding(
                        "current",
                        max(abs(value) for value in currents),
                        constraints.max_c_rate,
                        "Trajectory exceeds the maximum absolute C-rate",
                        hard=True,
                    )
                )
            if min(currents) < 0:
                findings.append(
                    self._finding(
                        "current_direction",
                        min(currents),
                        ">= 0 during charging",
                        "Trajectory contains a discharge current in a charge policy",
                        hard=True,
                    )
                )

        temperatures = arrays.get("temperature")
        if temperatures:
            maximum_temperature = max(temperatures)
            if maximum_temperature > constraints.max_temperature_c:
                findings.append(
                    self._finding(
                        "temperature",
                        maximum_temperature,
                        constraints.max_temperature_c,
                        "Trajectory exceeds the maximum temperature",
                        hard=True,
                    )
                )
            baseline = temperatures[0] if ambient_temperature_c is None else ambient_temperature_c
            if not isinstance(baseline, (int, float)) or not math.isfinite(float(baseline)):
                findings.append(
                    self._finding(
                        "ambient_temperature",
                        str(baseline),
                        "finite temperature",
                        "Ambient temperature is missing or non-finite",
                    )
                )
            else:
                rise = maximum_temperature - float(baseline)
                if rise > constraints.max_temperature_rise_c:
                    findings.append(
                        self._finding(
                            "temperature_rise",
                            rise,
                            constraints.max_temperature_rise_c,
                            "Trajectory exceeds the maximum temperature rise",
                            hard=True,
                        )
                    )

            times = arrays.get("time")
            if times and len(times) == len(temperatures):
                rates: list[float] = []
                for time_0, time_1, temp_0, temp_1 in zip(
                    times,
                    times[1:],
                    temperatures,
                    temperatures[1:],
                    strict=False,
                ):
                    dt = time_1 - time_0
                    if dt > 0:
                        rates.append((temp_1 - temp_0) / dt)
                if rates and max(rates) > constraints.max_temperature_rise_rate_c_per_min:
                    findings.append(
                        self._finding(
                            "temperature_rise_rate",
                            max(rates),
                            constraints.max_temperature_rise_rate_c_per_min,
                            "Trajectory heats faster than the allowed rate",
                            hard=True,
                        )
                    )

        soc_values = arrays.get("soc")
        if soc_values and (
            min(soc_values) < constraints.min_soc
            or max(soc_values) > constraints.max_soc
        ):
            findings.append(
                self._finding(
                    "soc_window",
                    f"[{min(soc_values)}, {max(soc_values)}]",
                    f"[{constraints.min_soc}, {constraints.max_soc}]",
                    "Trajectory SOC leaves the allowed window",
                    hard=True,
                )
            )

        margins = arrays.get("plating_margin")
        if margins and min(margins) < constraints.min_plating_margin:
            findings.append(
                self._finding(
                    "plating_margin",
                    min(margins),
                    constraints.min_plating_margin,
                    "Plating-risk proxy margin falls below its hard minimum",
                    hard=True,
                )
            )

    def _decide(self, policy_id: str, findings: Iterable[_Finding]) -> SafetyResult:
        materialized = list(findings)
        if any(finding.hard for finding in materialized):
            decision = SafetyDecision.REJECT
        elif materialized:
            decision = SafetyDecision.FALLBACK
        else:
            decision = SafetyDecision.ALLOW
        return self._result(
            decision,
            policy_id,
            [finding.violation for finding in materialized],
        )

    def _result(
        self,
        decision: SafetyDecision,
        policy_id: str,
        violations: list[ConstraintViolation],
    ) -> SafetyResult:
        return SafetyResult(
            decision=decision,
            policy_id=policy_id,
            violations=violations,
            fallback_policy_id=(
                self.fallback_policy_id if decision != SafetyDecision.ALLOW else None
            ),
            shield_version=self.version,
            safety_case_hash=self.safety_case_hash,
        )

    @staticmethod
    def _finding(
        constraint: str,
        value: Any,
        limit: Any,
        message: str,
        *,
        hard: bool = False,
    ) -> _Finding:
        safe_value = value if value is None or isinstance(value, (int, float, str)) else str(value)
        safe_limit = limit if limit is None or isinstance(limit, (int, float, str)) else str(limit)
        return _Finding(
            violation=ConstraintViolation(
                constraint=constraint,
                value=safe_value,
                limit=safe_limit,
                message=message,
            ),
            hard=hard,
        )

    def evaluate_many(
        self,
        pairs: Iterable[tuple[ChargingPolicy, SimulationTrajectory | None]],
        **context: Any,
    ) -> list[SafetyResult]:
        """Evaluate a sequence while preserving input order."""

        return [self.evaluate(policy, trajectory, **context) for policy, trajectory in pairs]


DeterministicSafetyShield = SafetyShield
