"""Charging policy, simulation trajectory, Pareto, and safety contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyFamily(StrEnum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    LIFE = "LIFE"
    FALLBACK = "FALLBACK"


class ChargingPolicy(StrictModel):
    policy_id: str
    family: PolicyFamily
    c_rates: list[float] = Field(min_length=1, max_length=6)
    soc_breaks: list[float] = Field(min_length=1, max_length=6)
    v_max: float = Field(gt=0)
    cv_cutoff_c: float = Field(gt=0)
    target_soc: float = Field(gt=0, le=1)
    personalized: bool = False

    @model_validator(mode="after")
    def validate_stages(self) -> ChargingPolicy:
        if len(self.c_rates) != len(self.soc_breaks):
            raise ValueError("c_rates and soc_breaks must have identical lengths")
        if any(rate <= 0 for rate in self.c_rates):
            raise ValueError("all C-rates must be positive")
        if self.soc_breaks != sorted(self.soc_breaks):
            raise ValueError("soc_breaks must be monotonically increasing")
        if self.soc_breaks[-1] != self.target_soc:
            raise ValueError("final SOC break must equal target_soc")
        return self


class SimulationMetrics(StrictModel):
    charge_time_min: float = Field(gt=0)
    degradation_proxy: float = Field(ge=0)
    max_temperature_c: float
    energy_loss_wh: float = Field(ge=0)
    feasibility: bool


class SimulationTrajectory(StrictModel):
    policy_id: str
    simulator_version: str
    status: str
    time_min: list[float]
    voltage_v: list[float]
    current_c: list[float]
    temperature_c: list[float]
    soc: list[float]
    plating_margin: list[float]
    metrics: SimulationMetrics | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> SimulationTrajectory:
        lengths = {
            len(self.time_min),
            len(self.voltage_v),
            len(self.current_c),
            len(self.temperature_c),
            len(self.soc),
            len(self.plating_margin),
        }
        if len(lengths) != 1:
            raise ValueError("all trajectory arrays must have equal length")
        if self.status == "SUCCESS" and (not self.time_min or self.metrics is None):
            raise ValueError("successful simulation requires samples and metrics")
        if self.status != "SUCCESS" and not self.error:
            raise ValueError("failed simulation requires an error message")
        return self


class ConstraintViolation(StrictModel):
    constraint: str
    value: float | str | None
    limit: float | str | None
    message: str


class SafetyDecision(StrEnum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    FALLBACK = "FALLBACK"


class SafetyResult(StrictModel):
    decision: SafetyDecision
    policy_id: str
    violations: list[ConstraintViolation] = Field(default_factory=list)
    fallback_policy_id: str | None = None
    shield_version: str
    safety_case_hash: str

    @model_validator(mode="after")
    def validate_decision(self) -> SafetyResult:
        if self.decision != SafetyDecision.ALLOW and not self.violations:
            raise ValueError("non-ALLOW decisions require violations")
        if self.decision == SafetyDecision.FALLBACK and not self.fallback_policy_id:
            raise ValueError("FALLBACK requires fallback_policy_id")
        return self


class EvaluatedPolicy(StrictModel):
    policy: ChargingPolicy
    trajectory: SimulationTrajectory
    safety: SafetyResult
    pareto_optimal: bool = False


class PolicyResponse(StrictModel):
    cell_id: str
    policies: list[EvaluatedPolicy]
    pareto_front: list[str]
    rejected: list[str]
    fallback: str | None
    evidence_ids: list[str] = Field(default_factory=list)
