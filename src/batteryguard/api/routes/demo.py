from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from batteryguard.api.dependencies import get_engine
from batteryguard.api.models import BlindRevealRequest
from batteryguard.demo.blind_pool import BlindCellNotFoundError
from batteryguard.demo.engine import DemoEngine
from batteryguard.demo.reveal import (
    AlreadyRevealedError,
    InvalidPredictionError,
    RevealAuthorizationError,
)

router = APIRouter(prefix="/v1/demo")


@router.post("/blind-reveal")
def blind_reveal(
    request: BlindRevealRequest, engine: DemoEngine = Depends(get_engine)
) -> dict[str, object]:
    try:
        return engine.reveal(request.cell_id, request.evaluator_token)
    except RevealAuthorizationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except BlindCellNotFoundError as exc:
        raise HTTPException(status_code=404, detail="blind cell not found") from exc
    except AlreadyRevealedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidPredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
