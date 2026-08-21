from __future__ import annotations

import hmac
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.encoders import jsonable_encoder

from batteryguard.demo.blind_pool import BlindPool
from batteryguard.demo.reveal import (
    AlreadyRevealedError,
    BlindRevealService,
    RevealAuthorizationError,
)
from batteryguard.evidence.ledger import EvidenceLedger
from batteryguard.schemas.data import CellRecord
from batteryguard.schemas.evidence import EvidenceStatus


def _pool() -> BlindPool:
    cells = [
        CellRecord(
            cell_id="hidden-01",
            chemistry="LFP_graphite",
            nominal_capacity_ah=1.1,
            batch_id="batch-a",
            protocol_id="hidden-protocol",
            cycle_life=947,
        ),
        CellRecord(
            cell_id="hidden-02",
            chemistry="LFP_graphite",
            nominal_capacity_ah=1.1,
            batch_id="batch-b",
            protocol_id="hidden-protocol",
            cycle_life=500,
        ),
    ]
    early = {
        "hidden-01": [
            {
                "cell_id": "hidden-01",
                "cycle_index": 1,
                "discharge_capacity_ah": 1.05,
                # Even a contaminated input row is scrubbed at the boundary.
                "cycle_life": 947,
                "nested": {"actual_cycle_life": 947},
            }
        ],
        "hidden-02": [],
    }
    return BlindPool(cells, early)


def _assert_no_lifetime(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True).lower()
    assert "cycle_life" not in serialized
    assert "actual_cycle_life" not in serialized


def test_every_public_pool_representation_hides_lifetime() -> None:
    pool = _pool()

    _assert_no_lifetime(pool.list_public_cells())
    _assert_no_lifetime(pool.get_public_cell("hidden-01"))
    _assert_no_lifetime(pool.random_public_cell(seed=42))
    _assert_no_lifetime(pool.model_dump())
    _assert_no_lifetime(pool.to_dict())
    _assert_no_lifetime(pool.to_json())
    _assert_no_lifetime(pool.json())
    _assert_no_lifetime(repr(pool))
    _assert_no_lifetime(dict(pool))
    _assert_no_lifetime(jsonable_encoder(pool))
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(pool)
    with pytest.raises(TypeError):
        vars(pool)


@pytest.mark.parametrize("token", [None, "", "wrong-token"])
def test_missing_or_wrong_token_is_rejected_and_audited(
    tmp_path: Path, token: str | None
) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    service = BlindRevealService(_pool(), ledger, "correct-token")

    with pytest.raises(RevealAuthorizationError):
        service.reveal("hidden-01", token, 920, 760, 1080)

    records = ledger.records()
    assert len(records) == 1
    assert records[0].event_type == "BLIND_REVEAL_ATTEMPT_REJECTED"
    assert records[0].status is EvidenceStatus.REJECTED
    assert records[0].payload["authorized"] is False
    serialized_record = json.dumps(records[0].model_dump(mode="json")).lower()
    assert "correct-token" not in serialized_record
    if token:
        assert token not in serialized_record
    _assert_no_lifetime(records[0].model_dump(mode="json"))


def test_token_comparison_uses_fixed_length_constant_time_digests(tmp_path: Path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    service = BlindRevealService(_pool(), ledger, "a-much-longer-evaluator-token")

    with patch(
        "batteryguard.demo.reveal.hmac.compare_digest",
        wraps=hmac.compare_digest,
    ) as compare_digest, pytest.raises(RevealAuthorizationError):
        service.reveal("hidden-01", "x", 920, 760, 1080)

    compared_candidate, compared_expected = compare_digest.call_args.args
    assert isinstance(compared_candidate, bytes)
    assert len(compared_candidate) == len(compared_expected) == 32


def test_authorized_reveal_appends_validation_to_existing_claim(tmp_path: Path) -> None:
    pool = _pool()
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    claim_id = "BG-CL-hidden-01-PRED-v1"
    pending = ledger.append(
        claim_id=claim_id,
        event_type="PREDICTION_CREATED",
        claim="Cell predicted to reach EOL at 920 cycles",
        status=EvidenceStatus.PENDING,
        data_version="fixture-v1",
        model_version="model-v1",
        payload={"prediction_interval": [760, 1080]},
    )
    service = BlindRevealService(
        pool,
        ledger,
        "correct-token",
        data_version="fixture-v1",
        model_version="model-v1",
    )

    result = service.reveal(
        "hidden-01", "correct-token", 920, 760, 1080, claim_id=claim_id
    )

    assert result["actual_cycle_life"] == 947
    assert result["actual"] == 947
    assert result["covered"] is True
    assert result["coverage"] is True
    assert result["absolute_error"] == 27
    assert result["evidence_status"] == "SUPPORTED_IN_THIS_TEST"
    assert pool.is_revealed("hidden-01")
    records = ledger.records_for_claim(claim_id)
    assert len(records) == 2
    assert records[0] == pending
    assert records[1].event_type == "BLIND_REVEAL_COMPLETED"
    assert records[1].payload["actual_cycle_life"] == 947
    assert ledger.verify_chain()


def test_uncovered_reveal_records_not_supported_status(tmp_path: Path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    service = BlindRevealService(_pool(), ledger, "correct-token")

    result = service.reveal("hidden-02", "correct-token", 450, 400, 480)

    assert result["covered"] is False
    assert result["absolute_error"] == 50
    assert result["status"] == "NOT_SUPPORTED_IN_THIS_TEST"


def test_only_one_concurrent_reveal_can_succeed(tmp_path: Path) -> None:
    pool = _pool()
    path = tmp_path / "evidence.jsonl"
    services = [
        BlindRevealService(pool, EvidenceLedger(path), "correct-token") for _ in range(8)
    ]

    def attempt(service: BlindRevealService) -> dict[str, Any] | type[Exception]:
        try:
            return service.reveal("hidden-01", "correct-token", 920, 760, 1080)
        except AlreadyRevealedError:
            return AlreadyRevealedError

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(attempt, services))

    successes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    assert len(successes) == 1
    assert outcomes.count(AlreadyRevealedError) == 7
    records = EvidenceLedger(path).records()
    assert sum(record.event_type == "BLIND_REVEAL_COMPLETED" for record in records) == 1
    assert sum(record.event_type == "BLIND_REVEAL_ATTEMPT_REJECTED" for record in records) == 7
    assert EvidenceLedger(path).verify_chain()


def test_pool_rejects_second_success_even_if_services_use_different_ledgers(
    tmp_path: Path,
) -> None:
    pool = _pool()
    first = BlindRevealService(pool, EvidenceLedger(tmp_path / "first.jsonl"), "token")
    second = BlindRevealService(pool, EvidenceLedger(tmp_path / "second.jsonl"), "token")

    first.reveal("hidden-01", "token", 920, 760, 1080)
    with pytest.raises(AlreadyRevealedError):
        second.reveal("hidden-01", "token", 920, 760, 1080)

    assert sum(
        record.event_type == "BLIND_REVEAL_COMPLETED"
        for path in (tmp_path / "first.jsonl", tmp_path / "second.jsonl")
        for record in EvidenceLedger(path).records()
    ) == 1


def test_ledger_rejects_second_success_from_recreated_pool(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    first = BlindRevealService(_pool(), EvidenceLedger(path), "token")
    recreated = BlindRevealService(_pool(), EvidenceLedger(path), "token")

    first.reveal("hidden-01", "token", 920, 760, 1080)
    with pytest.raises(AlreadyRevealedError):
        recreated.reveal("hidden-01", "token", 920, 760, 1080)

    records = EvidenceLedger(path).records()
    assert [record.event_type for record in records] == [
        "BLIND_REVEAL_COMPLETED",
        "BLIND_REVEAL_ATTEMPT_REJECTED",
    ]
    assert records[1].payload["reason"] == "ALREADY_REVEALED"
