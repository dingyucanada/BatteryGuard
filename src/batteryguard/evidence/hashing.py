"""Deterministic JSON serialization and SHA-256 helpers.

The evidence ledger hashes the JSON representation written to disk.  Keeping the
serialization rules in this small module makes it possible for independent tools
to reproduce a record hash without importing the ledger implementation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _json_value(value: Any) -> Any:
    """Convert supported Python values to deterministic JSON-compatible values."""

    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json", by_alias=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            serialized = value.isoformat()
        else:
            serialized = value.astimezone(UTC).isoformat()
            if serialized.endswith("+00:00"):
                serialized = f"{serialized[:-6]}Z"
        return serialized
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Return UTF-8 canonical JSON with sorted keys and no insignificant spaces.

    This intentionally implements the project's compact canonical form rather
    than claiming full RFC 8785 number normalization.  Inputs containing non-finite
    floats are rejected so that the same value cannot acquire platform-specific
    spellings.
    """

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return :func:`canonical_json` encoded as UTF-8."""

    return canonical_json(value).encode("utf-8")


def sha256_hex(value: bytes | bytearray | memoryview | str) -> str:
    """Return the lowercase SHA-256 hex digest of bytes or UTF-8 text."""

    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return sha256_hex(canonical_json_bytes(value))


def hash_record(record: BaseModel | Mapping[str, Any]) -> str:
    """Hash a ledger record, excluding its self-referential ``record_hash`` field."""

    if isinstance(record, BaseModel):
        payload = record.model_dump(mode="json", exclude={"record_hash"})
    else:
        payload = dict(record)
        payload.pop("record_hash", None)
    return canonical_sha256(payload)
