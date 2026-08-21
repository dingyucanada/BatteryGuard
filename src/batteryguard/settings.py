"""Typed runtime configuration with offline-safe defaults."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _ephemeral_reveal_token() -> str:
    """Return a process-local fallback that is never a published credential."""

    return secrets.token_urlsafe(32)


class SafetySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    v_max: float = 3.60
    c_rate_max: float = 4.0
    t_max_c: float = 45.0
    temperature_rise_max_c: float = 15.0
    thermal_rate_max_c_per_min: float = 2.0
    soc_min: float = 0.0
    soc_max: float = 0.80
    plating_margin_min: float = 0.05


class PredictionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    early_cycles: int = Field(default=30, ge=10, le=100)
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    ood_threshold: float = Field(default=0.98, gt=0.0, le=1.0)
    quality_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    interval_width_ratio_max: float = Field(default=0.45, gt=0.0)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_root: Path
    offline: bool = True
    seed: int = 42
    reveal_token: str = Field(default_factory=_ephemeral_reveal_token)
    prediction: PredictionSettings = PredictionSettings()
    safety: SafetySettings = SafetySettings()

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> AppSettings:
        root = project_root or Path.cwd()
        configured_reveal_token = os.getenv("BATTERYGUARD_REVEAL_TOKEN")
        return cls(
            project_root=root,
            offline=os.getenv("BATTERYGUARD_OFFLINE", "1") != "0",
            seed=int(os.getenv("BATTERYGUARD_SEED", "42")),
            reveal_token=configured_reveal_token or _ephemeral_reveal_token(),
        )
