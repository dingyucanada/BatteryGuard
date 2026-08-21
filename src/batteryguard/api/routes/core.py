from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from batteryguard.api.dependencies import get_engine
from batteryguard.api.models import HealthResponse
from batteryguard.demo.engine import DemoEngine

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(engine: DemoEngine = Depends(get_engine)) -> HealthResponse:
    return HealthResponse.model_validate(engine.health())


@router.get("/v1/evidence/{claim_id}")
def evidence(claim_id: str, engine: DemoEngine = Depends(get_engine)) -> list[dict[str, object]]:
    records = engine.evidence(claim_id)
    if not records:
        raise HTTPException(status_code=404, detail="claim not found")
    return records


@router.get("/v1/evidence")
def all_evidence(engine: DemoEngine = Depends(get_engine)) -> list[dict[str, object]]:
    return engine.evidence()


@router.get("/v1/models/report")
def model_report(engine: DemoEngine = Depends(get_engine)) -> dict[str, object]:
    return engine.model_report()
