"""Prediction, uncertainty, OOD, abstention, and diagnosis contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionResponse(StrictModel):
    cell_id: str
    observed_cycles: int = Field(gt=0)
    point_estimate: float = Field(gt=0)
    rul_estimate: float = Field(ge=0)
    interval_low: float = Field(ge=0)
    interval_high: float = Field(gt=0)
    coverage_target: float = Field(gt=0, lt=1)
    ood_score: float = Field(ge=0, le=1)
    abstain: bool
    abstention_reasons: list[str] = Field(default_factory=list)
    coverage_note: str
    model_name: str
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> PredictionResponse:
        if not self.interval_low <= self.point_estimate <= self.interval_high:
            raise ValueError("point estimate must lie inside prediction interval")
        if self.abstain and not self.abstention_reasons:
            raise ValueError("abstention must include at least one reason")
        return self


class RiskFingerprint(StrictModel):
    cell_id: str
    polarization_risk: float = Field(ge=0, le=1)
    thermal_stress: float = Field(ge=0, le=1)
    efficiency_instability: float = Field(ge=0, le=1)
    curve_shift: float = Field(ge=0, le=1)
    data_quality: float = Field(ge=0, le=1)
    ood_score: float = Field(ge=0, le=1)
    nearest_cells: list[str] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
