from __future__ import annotations

from fastapi import APIRouter, Depends

from batteryguard.api.dependencies import get_engine
from batteryguard.api.models import PolicyEvaluationRequest
from batteryguard.demo.engine import DemoEngine
from batteryguard.schemas.policy import EvaluatedPolicy

router = APIRouter(prefix="/v1/policies")


@router.post("/evaluate", response_model=EvaluatedPolicy)
def evaluate_policy(
    request: PolicyEvaluationRequest,
    engine: DemoEngine = Depends(get_engine),
) -> EvaluatedPolicy:
    return engine.evaluate_policy(
        request.policy,
        ambient_temperature_c=request.ambient_temperature_c,
        initial_soc=request.initial_soc,
        ood_score=request.ood_score,
        abstain=request.abstain,
    )
