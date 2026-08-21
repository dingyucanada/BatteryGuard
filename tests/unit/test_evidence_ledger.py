from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from batteryguard.evidence.ledger import (
    DuplicateEvidenceError,
    EvidenceLedger,
    LedgerIntegrityError,
)
from batteryguard.schemas.evidence import EvidenceStatus


def _append_prediction(
    ledger: EvidenceLedger,
    index: int,
    *,
    claim_id: str | None = None,
    event_type: str = "PREDICTION_CREATED",
) -> None:
    ledger.append(
        claim_id=claim_id or f"claim:{index}",
        event_type=event_type,
        claim=f"prediction {index}",
        status=EvidenceStatus.PENDING,
        data_version="test-v1",
        model_version="baseline-v1",
        payload={"index": index},
    )


def test_ledger_appends_a_verifiable_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "evidence.jsonl"
    ledger = EvidenceLedger(path)

    first = ledger.append(
        claim_id="claim:cell-1",
        event_type="PREDICTION_CREATED",
        claim="Cell reaches EOL in the prediction interval",
        status="PENDING",
        data_version="dataset-v1",
        payload={"prediction_interval": [90, 110]},
    )
    second = ledger.append(
        claim_id="claim:cell-1",
        event_type="BLIND_REVEAL_COMPLETED",
        claim="Cell reaches EOL in the prediction interval",
        status="SUPPORTED_IN_THIS_TEST",
        data_version="dataset-v1",
        payload={"actual_cycle_life": 100, "covered": True},
    )

    assert first.sequence == 1
    assert first.previous_hash is None
    assert len(first.record_hash) == 64
    assert second.sequence == 2
    assert second.previous_hash == first.record_hash
    assert ledger.verify_chain()
    assert EvidenceLedger(path).records() == [first, second]


def test_same_claim_reveal_is_appended_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    _append_prediction(ledger, 1, claim_id="claim:shared")
    before = path.read_bytes()
    _append_prediction(
        ledger,
        2,
        claim_id="claim:shared",
        event_type="BLIND_REVEAL_COMPLETED",
    )

    after = path.read_bytes()
    records = ledger.records_for_claim("claim:shared")
    assert after.startswith(before)
    assert len(records) == 2
    assert records[0].event_type == "PREDICTION_CREATED"
    assert records[1].event_type == "BLIND_REVEAL_COMPLETED"


def test_tampering_is_detected_and_blocks_future_appends(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    _append_prediction(ledger, 1)
    stored = json.loads(path.read_text().strip())
    stored["claim"] = "silently altered"
    path.write_text(json.dumps(stored) + "\n")

    assert not ledger.verify_chain()
    with pytest.raises(LedgerIntegrityError, match="record_hash mismatch"):
        ledger.verify_chain(raise_on_error=True)
    with pytest.raises(LedgerIntegrityError):
        _append_prediction(ledger, 2)


def test_clean_tail_truncation_is_detected_by_chain_head(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    _append_prediction(ledger, 1)
    _append_prediction(ledger, 2)
    lines = path.read_bytes().splitlines(keepends=True)

    path.write_bytes(lines[0])

    assert not ledger.verify_chain()
    with pytest.raises(LedgerIntegrityError, match="tail may have been truncated"):
        ledger.verify_chain(raise_on_error=True)
    with pytest.raises(LedgerIntegrityError):
        _append_prediction(ledger, 3)


def test_emptying_ledger_or_removing_chain_head_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    _append_prediction(ledger, 1)

    ledger.checkpoint_path.unlink()
    assert not ledger.verify_chain()
    with pytest.raises(LedgerIntegrityError, match="checkpoint is missing"):
        ledger.verify_chain(raise_on_error=True)

    # Restoring the checkpoint and emptying only the JSONL is another tail deletion.
    _append_prediction(EvidenceLedger(tmp_path / "other.jsonl"), 2)
    other = EvidenceLedger(tmp_path / "other.jsonl")
    other.path.write_bytes(b"")
    assert not other.verify_chain()
    with pytest.raises(LedgerIntegrityError, match="ledger was emptied"):
        other.verify_chain(raise_on_error=True)


def test_unique_key_is_atomic_and_reserved_payload_cannot_poison_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    first = ledger.append(
        claim_id="claim:one",
        event_type="UNIQUE_EVENT",
        claim="first",
        status=EvidenceStatus.PENDING,
        data_version="test-v1",
        unique_key="reveal:cell-1",
    )
    before_ledger = path.read_bytes()
    before_checkpoint = ledger.checkpoint_path.read_bytes()

    with pytest.raises(DuplicateEvidenceError):
        ledger.append(
            claim_id="claim:two",
            event_type="UNIQUE_EVENT",
            claim="duplicate",
            status=EvidenceStatus.PENDING,
            data_version="test-v1",
            unique_key="reveal:cell-1",
        )

    assert path.read_bytes() == before_ledger
    assert ledger.checkpoint_path.read_bytes() == before_checkpoint
    assert ledger.records() == [first]

    with pytest.raises(ValueError, match="reserved"):
        ledger.append(
            claim_id="claim:poison",
            event_type="FORGED_EVENT",
            claim="attempt to pre-claim a uniqueness key",
            status=EvidenceStatus.PENDING,
            data_version="test-v1",
            payload={"_ledger_unique_key": "reveal:cell-2"},
        )
    assert ledger.records() == [first]

    with pytest.raises(ValueError, match="non-empty string"):
        ledger.append(
            claim_id="claim:bad-key",
            event_type="UNIQUE_EVENT",
            claim="invalid uniqueness type",
            status=EvidenceStatus.PENDING,
            data_version="test-v1",
            unique_key=0,  # type: ignore[arg-type]
        )


def test_concurrent_duplicate_unique_key_has_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledgers = [EvidenceLedger(path) for _ in range(12)]

    def append_unique(index: int) -> bool:
        try:
            ledgers[index].append(
                claim_id=f"claim:{index}",
                event_type="BLIND_REVEAL_COMPLETED",
                claim="one successful reveal",
                status=EvidenceStatus.SUPPORTED,
                data_version="test-v1",
                unique_key="blind-reveal-success:shared-cell",
            )
        except DuplicateEvidenceError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(append_unique, range(12)))

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 11
    assert len(EvidenceLedger(path).records()) == 1
    assert EvidenceLedger(path).verify_chain()


def test_concurrent_ledger_instances_keep_sequence_contiguous(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledgers = [EvidenceLedger(path), EvidenceLedger(path)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_append_prediction, ledgers[index % 2], index)
            for index in range(24)
        ]
        for future in futures:
            future.result()

    records = EvidenceLedger(path).records()
    assert [record.sequence for record in records] == list(range(1, 25))
    assert len({record.record_hash for record in records}) == 24
    assert EvidenceLedger(path).verify_chain()
    checkpoint = json.loads(EvidenceLedger(path).checkpoint_path.read_text())
    assert checkpoint == {
        "record_hash": records[-1].record_hash,
        "sequence": records[-1].sequence,
    }
