from __future__ import annotations

import pytest
from pydantic import ValidationError

from batteryguard.schemas.data import CellRecord
from batteryguard.schemas.policy import (
    ChargingPolicy,
    PolicyFamily,
    SafetyDecision,
    SafetyResult,
)
from batteryguard.schemas.prediction import PredictionResponse


def test_censored_cell_cannot_fabricate_lifetime() -> None:
    with pytest.raises(ValidationError, match="must not fabricate"):
        CellRecord(
            cell_id="C-1",
            chemistry="LFP_graphite",
            nominal_capacity_ah=1.1,
            protocol_id="P-1",
            cycle_life=900,
            censored=True,
        )


def test_policy_stage_contract_is_strict() -> None:
    with pytest.raises(ValidationError, match="identical lengths"):
        ChargingPolicy(
            policy_id="bad",
            family=PolicyFamily.FAST,
            c_rates=[3.0, 2.0],
            soc_breaks=[0.8],
            v_max=3.6,
            cv_cutoff_c=0.08,
            target_soc=0.8,
        )


def test_prediction_requires_reason_when_abstaining() -> None:
    with pytest.raises(ValidationError, match="at least one reason"):
        PredictionResponse(
            cell_id="C-1",
            observed_cycles=30,
            point_estimate=900,
            rul_estimate=870,
            interval_low=700,
            interval_high=1100,
            coverage_target=0.9,
            ood_score=0.99,
            abstain=True,
            coverage_note="held-out calibration",
            model_name="xgboost",
            evidence_ids=["claim-1"],
        )


def test_non_allow_safety_result_requires_violation() -> None:
    with pytest.raises(ValidationError, match="require violations"):
        SafetyResult(
            decision=SafetyDecision.REJECT,
            policy_id="p-1",
            shield_version="safety-1",
            safety_case_hash="abc",
        )
