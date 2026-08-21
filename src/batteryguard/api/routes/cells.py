from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from batteryguard.api.dependencies import get_engine
from batteryguard.api.models import PolicyRequest, PredictRequest
from batteryguard.demo.blind_pool import BlindCellNotFoundError
from batteryguard.demo.engine import DemoEngine
from batteryguard.schemas.policy import PolicyResponse
from batteryguard.schemas.prediction import PredictionResponse, RiskFingerprint

router = APIRouter(prefix="/v1/cells")


@router.get("/blind")
def blind_cells(engine: DemoEngine = Depends(get_engine)) -> list[dict[str, object]]:
    return engine.list_blind_cells()


@router.get("/{cell_id}/early-cycles")
def early_cycles(cell_id: str, engine: DemoEngine = Depends(get_engine)) -> dict[str, object]:
    try:
        return engine.public_cell(cell_id)
    except BlindCellNotFoundError as exc:
        raise HTTPException(status_code=404, detail="blind cell not found") from exc


@router.post("/{cell_id}/predict", response_model=PredictionResponse)
def predict(
    cell_id: str,
    request: PredictRequest | None = None,
    engine: DemoEngine = Depends(get_engine),
) -> PredictionResponse:
    try:
        return engine.predict(
            cell_id,
            quality_override=request.quality_override if request is not None else None,
        )
    except BlindCellNotFoundError as exc:
        raise HTTPException(status_code=404, detail="blind cell not found") from exc


@router.post("/{cell_id}/diagnose", response_model=RiskFingerprint)
def diagnose(cell_id: str, engine: DemoEngine = Depends(get_engine)) -> RiskFingerprint:
    try:
        return engine.diagnose(cell_id)
    except BlindCellNotFoundError as exc:
        raise HTTPException(status_code=404, detail="blind cell not found") from exc


@router.post("/{cell_id}/policies", response_model=PolicyResponse)
def policies(
    cell_id: str,
    request: PolicyRequest | None = None,
    engine: DemoEngine = Depends(get_engine),
) -> PolicyResponse:
    resolved = request or PolicyRequest()
    try:
        return engine.policies(
            cell_id,
            ambient_temperature_c=resolved.ambient_temperature_c,
            initial_soc=resolved.initial_soc,
        )
    except BlindCellNotFoundError as exc:
        raise HTTPException(status_code=404, detail="blind cell not found") from exc
