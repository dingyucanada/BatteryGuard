"""Evaluator-authorized, auditable blind lifetime reveal."""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
from typing import Any

from batteryguard.demo.blind_pool import (
    _REVEAL_CAPABILITY,
    BlindCellNotFoundError,
    BlindPool,
    CellAlreadyRevealedError,
)
from batteryguard.evidence.ledger import DuplicateEvidenceError, EvidenceLedger
from batteryguard.schemas.evidence import EvidenceStatus


class RevealError(RuntimeError):
    """Base exception for reveal failures."""


class RevealAuthorizationError(RevealError, PermissionError):
    """Raised for a missing or incorrect evaluator token."""


class AlreadyRevealedError(RevealError):
    """Raised when a cell has already had a successful reveal event."""


class InvalidPredictionError(RevealError, ValueError):
    """Raised when prediction values cannot support a meaningful comparison."""


class BlindRevealService:
    """Authorize reveals, compute validation metrics, and append audit evidence.

    A cell permits exactly one successful reveal.  Repeated authorized requests are
    rejected rather than returning the hidden value again.  The persistent ledger
    uniqueness key enforces this across service instances sharing the same ledger.
    """

    def __init__(
        self,
        pool: BlindPool,
        ledger: EvidenceLedger,
        evaluator_token: str,
        *,
        data_version: str = "blind-demo-v1",
        feature_version: str | None = None,
        model_version: str | None = None,
        split_id: str | None = "demo_hidden",
    ) -> None:
        if not isinstance(evaluator_token, str) or not evaluator_token:
            raise ValueError("evaluator_token must be a non-empty string")
        self._pool = pool
        self._ledger = ledger
        self._token_digest = self._digest_token(evaluator_token)
        self._data_version = data_version
        self._feature_version = feature_version
        self._model_version = model_version
        self._split_id = split_id
        self._lock = threading.RLock()

    @staticmethod
    def _digest_token(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8")).digest()

    def _token_is_valid(self, token: object) -> bool:
        # Always compare two fixed-length digests.  This avoids leaking the expected
        # token's length while preserving hmac.compare_digest's constant-time path.
        candidate = token if isinstance(token, str) else ""
        candidate_digest = self._digest_token(candidate)
        digest_matches = hmac.compare_digest(candidate_digest, self._token_digest)
        return isinstance(token, str) and bool(candidate) and digest_matches

    @staticmethod
    def _claim_id(cell_id: str) -> str:
        return f"claim:blind:{cell_id}:lifetime"

    def _record_rejection(
        self,
        cell_id: str,
        reason: str,
        *,
        claim_id: str | None = None,
    ) -> None:
        self._ledger.append(
            claim_id=claim_id or self._claim_id(cell_id),
            event_type="BLIND_REVEAL_ATTEMPT_REJECTED",
            claim=f"Blind lifetime reveal for cell {cell_id}",
            status=EvidenceStatus.REJECTED,
            data_version=self._data_version,
            feature_version=self._feature_version,
            model_version=self._model_version,
            split_id=self._split_id,
            safety_result="REJECT",
            payload={
                "cell_id": cell_id,
                "authorized": False,
                "reason": reason,
                "validation": "BLIND_REVEAL_REJECTED",
            },
        )

    def authorize(
        self,
        cell_id: str,
        token: str | None,
        *,
        claim_id: str | None = None,
    ) -> None:
        """Authenticate and audit failures without consulting blind-cell state.

        Keeping authorization independent from cell and prediction lookup prevents
        an unauthenticated caller from using error differences as an existence or
        workflow-state oracle.
        """

        if self._token_is_valid(token):
            return
        reason = "MISSING_TOKEN" if token is None or token == "" else "INVALID_TOKEN"
        self._record_rejection(cell_id, reason, claim_id=claim_id)
        raise RevealAuthorizationError("valid evaluator token required")

    @staticmethod
    def _prediction_values(
        point_estimate: float | int | None,
        interval_low: float | int | None,
        interval_high: float | int | None,
    ) -> tuple[float, float, float]:
        supplied = (point_estimate, interval_low, interval_high)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in supplied):
            raise InvalidPredictionError("prediction and interval values must be numeric")
        point, low, high = (float(value) for value in supplied)  # type: ignore[arg-type]
        if not all(math.isfinite(value) for value in (point, low, high)):
            raise InvalidPredictionError("prediction and interval values must be finite")
        if point <= 0 or low < 0 or high <= 0:
            raise InvalidPredictionError("prediction values must be positive")
        if low > high:
            raise InvalidPredictionError("interval_low must not exceed interval_high")
        if not low <= point <= high:
            raise InvalidPredictionError("point_estimate must lie inside the interval")
        return point, low, high

    def reveal(
        self,
        cell_id: str,
        token: str | None = None,
        point_estimate: float | int | None = None,
        interval_low: float | int | None = None,
        interval_high: float | int | None = None,
        *,
        claim_id: str | None = None,
    ) -> dict[str, Any]:
        """Reveal once with a valid token and return JSON-serializable validation.

        Missing and invalid tokens are rejected before cell lookup or prediction
        validation and are recorded without ever persisting the supplied token.
        """

        effective_claim_id = claim_id or self._claim_id(cell_id)
        self.authorize(cell_id, token, claim_id=effective_claim_id)

        with self._lock:
            if not self._pool.has_cell(cell_id):
                self._record_rejection(
                    cell_id, "CELL_NOT_IN_BLIND_POOL", claim_id=effective_claim_id
                )
                raise BlindCellNotFoundError(cell_id)
            try:
                point, low, high = self._prediction_values(
                    point_estimate, interval_low, interval_high
                )
            except InvalidPredictionError:
                self._record_rejection(
                    cell_id, "INVALID_PREDICTION", claim_id=effective_claim_id
                )
                raise

            try:
                actual = self._pool._begin_reveal(cell_id, _REVEAL_CAPABILITY)
            except CellAlreadyRevealedError as exc:
                self._record_rejection(
                    cell_id, "ALREADY_REVEALED", claim_id=effective_claim_id
                )
                raise AlreadyRevealedError(
                    f"cell {cell_id!r} has already been successfully revealed"
                ) from exc
            covered = low <= actual <= high
            absolute_error = abs(float(actual) - point)
            status = (
                EvidenceStatus.SUPPORTED if covered else EvidenceStatus.NOT_SUPPORTED
            )
            payload = {
                "cell_id": cell_id,
                "actual_cycle_life": actual,
                "actual": actual,
                "covered": covered,
                "coverage": covered,
                "absolute_error": absolute_error,
                "point_estimate": point,
                "prediction_interval": [low, high],
                "authorized": True,
                "validation": "BLIND_REVEAL_COMPLETED",
            }
            append_succeeded = False
            try:
                evidence = self._ledger.append(
                    claim_id=effective_claim_id,
                    event_type="BLIND_REVEAL_COMPLETED",
                    claim=f"Blind lifetime validation for cell {cell_id}",
                    status=status,
                    data_version=self._data_version,
                    feature_version=self._feature_version,
                    model_version=self._model_version,
                    split_id=self._split_id,
                    payload=payload,
                    unique_key=f"blind-reveal-success:{cell_id}",
                )
                append_succeeded = True
            except DuplicateEvidenceError as exc:
                self._record_rejection(
                    cell_id, "ALREADY_REVEALED", claim_id=effective_claim_id
                )
                raise AlreadyRevealedError(
                    f"cell {cell_id!r} has already been successfully revealed"
                ) from exc
            finally:
                self._pool._finish_reveal(
                    cell_id, _REVEAL_CAPABILITY, success=append_succeeded
                )

            return {
                **payload,
                "status": status.value,
                "evidence_status": status.value,
                "claim_id": evidence.claim_id,
                "sequence": evidence.sequence,
                "record_hash": evidence.record_hash,
            }


# Concise alias for application code that treats reveal as a domain service.
RevealService = BlindRevealService


__all__ = [
    "AlreadyRevealedError",
    "BlindRevealService",
    "InvalidPredictionError",
    "RevealAuthorizationError",
    "RevealError",
    "RevealService",
]
