"""Append-only evidence ledger record contract."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(StrEnum):
    PENDING = "PENDING"
    SUPPORTED = "SUPPORTED_IN_THIS_TEST"
    NOT_SUPPORTED = "NOT_SUPPORTED_IN_THIS_TEST"
    REJECTED = "REJECTED_BY_SAFETY"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime
    event_type: str
    claim: str
    status: EvidenceStatus
    data_version: str
    feature_version: str | None = None
    model_version: str | None = None
    split_id: str | None = None
    simulation_version: str | None = None
    policy_version: str | None = None
    safety_result: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str | None = None
    record_hash: str
