"""HTTP request and small operational response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from batteryguard.schemas.policy import ChargingPolicy


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetIngestRequest(StrictRequest):
    source_path: str = Field(
        description="Dataset directory relative to the server's configured data/raw root"
    )
    dataset_id: str = "matr-v1"


class SplitBuildRequest(StrictRequest):
    dataset_id: str = "demo-synthetic-v1"
    strategy: str = "cell_grouped"
    seed: int = 42


class TrainRequest(StrictRequest):
    model: str = "xgboost"
    early_cycles: int = Field(default=30, ge=10, le=100)


class PredictRequest(StrictRequest):
    quality_override: float | None = Field(default=None, ge=0, le=1)


class PolicyRequest(StrictRequest):
    ambient_temperature_c: float = Field(default=25.0, ge=-20, le=80)
    initial_soc: float = Field(default=0.10, ge=0, lt=1)


class PolicyEvaluationRequest(StrictRequest):
    policy: ChargingPolicy
    ambient_temperature_c: float = Field(default=25.0, ge=-20, le=80)
    initial_soc: float = Field(default=0.10, ge=0, lt=1)
    ood_score: float = Field(default=0.0, ge=0, le=1)
    abstain: bool = False


class BlindRevealRequest(StrictRequest):
    cell_id: str
    evaluator_token: str | None = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    software_version: str
    research_only: bool
    evidence_chain_valid: bool


class OperationalResponse(BaseModel):
    status: str
    details: dict[str, Any] = Field(default_factory=dict)
