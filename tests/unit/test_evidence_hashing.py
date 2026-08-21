from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from batteryguard.evidence.hashing import canonical_json, canonical_sha256, sha256_hex


def test_canonical_json_is_stable_across_mapping_order() -> None:
    left = {"z": [3, 2, 1], "a": {"é": True, "none": None}}
    right = {"a": {"none": None, "é": True}, "z": [3, 2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":{"none":null,"é":true},"z":[3,2,1]}'
    assert canonical_sha256(left) == canonical_sha256(right)


def test_sha256_and_datetime_serialization_are_reproducible() -> None:
    text = "BatteryGuard"
    expected = hashlib.sha256(text.encode()).hexdigest()

    assert sha256_hex(text) == expected
    assert sha256_hex(text.encode()) == expected
    assert canonical_json({"at": datetime(2026, 8, 21, tzinfo=UTC)}) == (
        '{"at":"2026-08-21T00:00:00Z"}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json({"value": value})
