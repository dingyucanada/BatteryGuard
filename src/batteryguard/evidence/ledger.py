"""Append-only, hash-chained JSONL evidence ledger."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from batteryguard.evidence.hashing import canonical_json, hash_record
from batteryguard.schemas.evidence import EvidenceRecord, EvidenceStatus

try:  # pragma: no cover - the project targets Unix, but keep imports portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class EvidenceLedgerError(RuntimeError):
    """Base exception for ledger failures."""


class LedgerIntegrityError(EvidenceLedgerError):
    """Raised when a JSONL ledger is malformed or its hash chain is invalid."""


class DuplicateEvidenceError(EvidenceLedgerError):
    """Raised when an atomic append uses a uniqueness key already in the ledger."""


_UNIQUE_KEY = "_ledger_unique_key"
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _shared_path_lock(path: Path) -> threading.RLock:
    key = str(path.absolute())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


class EvidenceLedger:
    """A durable JSONL ledger with a verifiable SHA-256 hash chain.

    Appends use an in-process re-entrant lock and, on Unix, an advisory file lock.
    Every append is a single ``os.write`` to a descriptor opened with ``O_APPEND``
    and followed by ``fsync``.  Existing records are never rewritten.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.checkpoint_path = self.path.with_name(f"{self.path.name}.head")
        self._thread_lock = _shared_path_lock(self.path)

    @contextmanager
    def _locked_file(self) -> Iterator[BinaryIO]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield handle
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @staticmethod
    def _read_locked(handle: BinaryIO) -> list[EvidenceRecord]:
        handle.seek(0)
        raw = handle.read()
        if not raw:
            return []
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LedgerIntegrityError("ledger is not valid UTF-8") from exc

        records: list[EvidenceRecord] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise LedgerIntegrityError(f"blank JSONL record at line {line_number}")
            try:
                value = json.loads(line)
                records.append(EvidenceRecord.model_validate(value))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise LedgerIntegrityError(
                    f"invalid evidence record at line {line_number}: {exc}"
                ) from exc
        return records

    @staticmethod
    def _assert_chain(records: list[EvidenceRecord]) -> None:
        expected_previous: str | None = None
        for expected_sequence, record in enumerate(records, start=1):
            if record.sequence != expected_sequence:
                raise LedgerIntegrityError(
                    f"sequence mismatch at record {expected_sequence}: "
                    f"found {record.sequence}"
                )
            if record.previous_hash != expected_previous:
                raise LedgerIntegrityError(
                    f"previous_hash mismatch at sequence {record.sequence}"
                )
            calculated = hash_record(record)
            if record.record_hash != calculated:
                raise LedgerIntegrityError(
                    f"record_hash mismatch at sequence {record.sequence}"
                )
            expected_previous = record.record_hash

    def _assert_checkpoint(self, records: list[EvidenceRecord]) -> None:
        """Check the external chain head used to detect clean tail truncation."""

        try:
            raw = self.checkpoint_path.read_bytes()
        except FileNotFoundError as exc:
            if records:
                raise LedgerIntegrityError(
                    "chain-head checkpoint is missing for a non-empty ledger"
                ) from exc
            return
        except OSError as exc:
            raise LedgerIntegrityError("chain-head checkpoint cannot be read") from exc
        try:
            checkpoint = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerIntegrityError("chain-head checkpoint is malformed") from exc
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "record_hash",
            "sequence",
        }:
            raise LedgerIntegrityError("chain-head checkpoint has an invalid shape")
        sequence = checkpoint["sequence"]
        record_hash = checkpoint["record_hash"]
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or not isinstance(record_hash, str)
            or len(record_hash) != 64
            or any(character not in "0123456789abcdef" for character in record_hash)
        ):
            raise LedgerIntegrityError("chain-head checkpoint has invalid values")
        if not records:
            raise LedgerIntegrityError("ledger was emptied after evidence had been recorded")
        head = records[-1]
        if sequence != head.sequence or record_hash != head.record_hash:
            raise LedgerIntegrityError(
                "chain-head checkpoint mismatch; ledger tail may have been truncated"
            )

    def _write_checkpoint(self, record: EvidenceRecord) -> None:
        """Atomically persist the current head after the JSONL append is durable."""

        checkpoint = (
            canonical_json(
                {"record_hash": record.record_hash, "sequence": record.sequence}
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.checkpoint_path.name}.", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                pending = memoryview(checkpoint)
                while pending:
                    written = os.write(handle.fileno(), pending)
                    if written <= 0:
                        raise EvidenceLedgerError(
                            "short write while updating chain-head checkpoint"
                        )
                    pending = pending[written:]
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.checkpoint_path)
            try:
                directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            except OSError:  # pragma: no cover - filesystem-dependent hardening.
                return
            try:
                os.fsync(directory_descriptor)
            except OSError:  # pragma: no cover - not all filesystems fsync directories.
                pass
            finally:
                os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

    def append(
        self,
        *,
        claim_id: str,
        event_type: str,
        claim: str,
        status: EvidenceStatus | str,
        data_version: str,
        payload: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
        feature_version: str | None = None,
        model_version: str | None = None,
        split_id: str | None = None,
        simulation_version: str | None = None,
        policy_version: str | None = None,
        safety_result: str | None = None,
        unique_key: str | None = None,
    ) -> EvidenceRecord:
        """Atomically append an event and return the persisted record.

        ``unique_key`` is optional.  When supplied, checking and appending occur
        under the same file lock.  This is used by blind reveal to guarantee at
        most one successful event per cell, including across service instances.
        """

        event_payload = dict(payload or {})
        if _UNIQUE_KEY in event_payload:
            raise ValueError(f"payload field {_UNIQUE_KEY!r} is reserved")
        if unique_key is not None:
            if not isinstance(unique_key, str) or not unique_key:
                raise ValueError("unique_key must be a non-empty string")
            event_payload[_UNIQUE_KEY] = unique_key

        with self._thread_lock, self._locked_file() as handle:
            records = self._read_locked(handle)
            self._assert_chain(records)
            self._assert_checkpoint(records)
            if unique_key is not None and any(
                record.payload.get(_UNIQUE_KEY) == unique_key for record in records
            ):
                raise DuplicateEvidenceError(
                    f"an evidence event with unique key {unique_key!r} already exists"
                )

            previous_hash = records[-1].record_hash if records else None
            draft = EvidenceRecord.model_validate(
                {
                    "claim_id": claim_id,
                    "sequence": len(records) + 1,
                    "timestamp": timestamp or datetime.now(UTC),
                    "event_type": event_type,
                    "claim": claim,
                    "status": status,
                    "data_version": data_version,
                    "feature_version": feature_version,
                    "model_version": model_version,
                    "split_id": split_id,
                    "simulation_version": simulation_version,
                    "policy_version": policy_version,
                    "safety_result": safety_result,
                    "payload": event_payload,
                    "previous_hash": previous_hash,
                    "record_hash": "0" * 64,
                }
            )
            record = draft.model_copy(update={"record_hash": hash_record(draft)})
            serialized = (canonical_json(record.model_dump(mode="json")) + "\n").encode(
                "utf-8"
            )
            end_position = handle.seek(0, os.SEEK_END)
            separator = b""
            if end_position:
                handle.seek(-1, os.SEEK_END)
                separator = b"" if handle.read(1) == b"\n" else b"\n"
            pending = memoryview(separator + serialized)
            while pending:
                written = os.write(handle.fileno(), pending)
                if written <= 0:
                    raise EvidenceLedgerError("short write while appending evidence record")
                pending = pending[written:]
            os.fsync(handle.fileno())
            self._write_checkpoint(record)
            return record

    add = append

    def append_record(
        self,
        record: EvidenceRecord | Mapping[str, Any],
        *,
        unique_key: str | None = None,
    ) -> EvidenceRecord:
        """Append event fields from a record while assigning new chain metadata."""

        data = (
            record.model_dump(mode="python")
            if isinstance(record, EvidenceRecord)
            else dict(record)
        )
        for generated_field in ("sequence", "previous_hash", "record_hash"):
            data.pop(generated_field, None)
        allowed = {
            "claim_id",
            "timestamp",
            "event_type",
            "claim",
            "status",
            "data_version",
            "feature_version",
            "model_version",
            "split_id",
            "simulation_version",
            "policy_version",
            "safety_result",
            "payload",
        }
        unexpected = set(data) - allowed
        if unexpected:
            raise ValueError(f"unsupported evidence fields: {sorted(unexpected)}")
        return self.append(**data, unique_key=unique_key)

    def records(self, *, verify: bool = True) -> list[EvidenceRecord]:
        """Return all records in file order as an isolated list."""

        with self._thread_lock, self._locked_file() as handle:
            records = self._read_locked(handle)
            if verify:
                self._assert_chain(records)
                self._assert_checkpoint(records)
            return records

    def records_for_claim(self, claim_id: str) -> list[EvidenceRecord]:
        """Return every version/event for a claim, preserving append order."""

        return [record for record in self.records() if record.claim_id == claim_id]

    def get_claim(self, claim_id: str) -> list[EvidenceRecord]:
        """Alias suited to the evidence lookup API; records are never collapsed."""

        return self.records_for_claim(claim_id)

    get = get_claim

    def verify_chain(self, *, raise_on_error: bool = False) -> bool:
        """Return whether sequence numbers, links, and hashes are all valid."""

        try:
            self.records(verify=True)
        except LedgerIntegrityError:
            if raise_on_error:
                raise
            return False
        return True

    def __iter__(self) -> Iterator[EvidenceRecord]:
        return iter(self.records())

    def __len__(self) -> int:
        return len(self.records())
