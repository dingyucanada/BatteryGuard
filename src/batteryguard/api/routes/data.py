from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from batteryguard.api.dependencies import get_engine
from batteryguard.api.models import (
    DatasetIngestRequest,
    OperationalResponse,
    SplitBuildRequest,
    TrainRequest,
)
from batteryguard.demo.engine import DemoEngine

router = APIRouter(prefix="/v1")


@router.post("/datasets/ingest", response_model=OperationalResponse)
def ingest(
    request: DatasetIngestRequest, engine: DemoEngine = Depends(get_engine)
) -> OperationalResponse:
    try:
        details = engine.ingest_dataset(request.source_path, request.dataset_id)
    except (OSError, ValueError) as exc:
        raw_root = str((engine.settings.project_root / "data" / "raw").resolve())
        detail = str(exc).replace(raw_root, "data/raw")
        raise HTTPException(status_code=422, detail=detail) from exc
    return OperationalResponse(status="INGESTED", details=details)


@router.get("/datasets/{dataset_id}/quality")
def quality(dataset_id: str, engine: DemoEngine = Depends(get_engine)) -> dict[str, object]:
    if dataset_id != engine.dataset_id:
        raise HTTPException(status_code=404, detail="dataset not loaded")
    return engine.quality_report()


@router.post("/splits/build", response_model=OperationalResponse)
def build_splits(
    request: SplitBuildRequest, engine: DemoEngine = Depends(get_engine)
) -> OperationalResponse:
    try:
        details = engine.build_splits(strategy=request.strategy, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return OperationalResponse(status="BUILT", details=details)


@router.post("/models/train", response_model=OperationalResponse)
def train(
    request: TrainRequest, engine: DemoEngine = Depends(get_engine)
) -> OperationalResponse:
    try:
        details = engine.retrain(model_name=request.model, early_cycles=request.early_cycles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return OperationalResponse(status="TRAINED", details=details)
