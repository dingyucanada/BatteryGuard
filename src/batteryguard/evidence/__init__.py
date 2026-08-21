"""Evidence hashing and append-only ledger public API."""

from batteryguard.evidence.hashing import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    hash_record,
    sha256_hex,
)
from batteryguard.evidence.ledger import (
    DuplicateEvidenceError,
    EvidenceLedger,
    EvidenceLedgerError,
    LedgerIntegrityError,
)

__all__ = [
    "DuplicateEvidenceError",
    "EvidenceLedger",
    "EvidenceLedgerError",
    "LedgerIntegrityError",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "hash_record",
    "sha256_hex",
]
