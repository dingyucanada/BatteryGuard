"""Canonical cell, cycle, timeseries, split, and quality schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitName(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"
    DEMO_HIDDEN = "demo_hidden"
    EXTERNAL_OOD = "external_ood"


class CellRecord(StrictModel):
    cell_id: str = Field(min_length=1)
    chemistry: str = Field(min_length=1)
    nominal_capacity_ah: float = Field(gt=0)
    batch_id: str | None = None
    protocol_id: str = Field(min_length=1)
    cycle_life: int | None = Field(default=None, gt=0)
    eol_threshold: float = Field(default=0.80, gt=0.0, lt=1.0)
    censored: bool = False

    @model_validator(mode="after")
    def validate_censoring(self) -> CellRecord:
        if self.censored and self.cycle_life is not None:
            raise ValueError("censored cells must not fabricate cycle_life")
        return self


class CycleRecord(StrictModel):
    cell_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=1)
    charge_capacity_ah: float = Field(gt=0)
    discharge_capacity_ah: float = Field(gt=0)
    coulombic_efficiency: float = Field(gt=0, le=1.2)
    charge_time_s: float = Field(gt=0)
    discharge_time_s: float = Field(gt=0)
    dcir_ohm: float | None = Field(default=None, gt=0)
    avg_temp_c: float | None = None
    max_temp_c: float | None = None
    charge_energy_wh: float | None = Field(default=None, gt=0)
    discharge_energy_wh: float | None = Field(default=None, gt=0)


class TimeSeriesPoint(StrictModel):
    cell_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=1)
    sample_index: int = Field(ge=0)
    time_s: float = Field(ge=0)
    current_a: float
    voltage_v: float = Field(gt=0)
    charge_capacity_ah: float = Field(ge=0)
    discharge_capacity_ah: float = Field(ge=0)
    temperature_c: float | None = None
    step_type: str | None = None


class SplitRecord(StrictModel):
    cell_id: str = Field(min_length=1)
    split: SplitName
    split_reason: str = Field(min_length=1)
    group_keys: dict[str, Any] = Field(default_factory=dict)
    ood_type: str | None = None


class DataQualityIssue(StrictModel):
    code: str
    severity: str
    message: str
    cell_id: str | None = None
    cycle_index: int | None = None


class DataQualityReport(StrictModel):
    dataset_id: str
    quality_score: float = Field(ge=0.0, le=1.0)
    valid_cells: int = Field(ge=0)
    valid_cycles: int = Field(ge=0)
    issues: list[DataQualityIssue] = Field(default_factory=list)
    leakage_checks_passed: bool = False
