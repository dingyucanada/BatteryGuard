"""Immutable constraints and reproducible safety-case hashing."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from batteryguard.constants import SAFETY_VERSION


@dataclass(frozen=True, slots=True)
class SafetyConstraints:
    """Hard limits enforced by :class:`batteryguard.safety.SafetyShield`.

    Values are deliberately explicit and conservative for the research demo.
    They are not certification limits and must not be used to control hardware.
    """

    min_voltage_v: float = 2.50
    max_voltage_v: float = 3.60
    max_c_rate: float = 4.00
    max_temperature_c: float = 45.0
    max_temperature_rise_c: float = 15.0
    max_temperature_rise_rate_c_per_min: float = 2.0
    min_soc: float = 0.0
    max_soc: float = 0.80
    min_plating_margin: float = 0.05
    ood_threshold: float = 0.70
    aggressive_c_rate: float = 1.50

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(isinstance(value, (int, float)) for value in values.values()):
            raise TypeError("all safety constraints must be numeric")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all safety constraints must be finite")
        if self.min_voltage_v <= 0 or self.min_voltage_v >= self.max_voltage_v:
            raise ValueError("voltage limits must be positive and ordered")
        if self.max_c_rate <= 0:
            raise ValueError("max_c_rate must be positive")
        if self.max_temperature_rise_c < 0:
            raise ValueError("max_temperature_rise_c cannot be negative")
        if self.max_temperature_rise_rate_c_per_min <= 0:
            raise ValueError("temperature-rise rate limit must be positive")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError("SOC limits must satisfy 0 <= min < max <= 1")
        if not 0 <= self.ood_threshold <= 1:
            raise ValueError("ood_threshold must lie in [0, 1]")
        if self.aggressive_c_rate <= 0:
            raise ValueError("aggressive_c_rate must be positive")

    def canonical_payload(self) -> dict[str, Any]:
        """Return every enforced value plus the shield implementation version."""

        return {"shield_version": SAFETY_VERSION, "constraints": asdict(self)}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SafetyConstraints:
        """Load either canonical names or the public YAML/settings aliases."""

        aliases = {
            "v_min": "min_voltage_v",
            "v_max": "max_voltage_v",
            "c_rate_max": "max_c_rate",
            "t_max_c": "max_temperature_c",
            "temperature_rise_max_c": "max_temperature_rise_c",
            "thermal_rate_max_c_per_min": "max_temperature_rise_rate_c_per_min",
            "soc_min": "min_soc",
            "soc_max": "max_soc",
            "plating_margin_min": "min_plating_margin",
        }
        field_names = set(cls.__dataclass_fields__)
        normalized: dict[str, Any] = {}
        for name, value in values.items():
            canonical_name = aliases.get(name, name)
            if canonical_name in field_names:
                normalized[canonical_name] = value
        return cls(**normalized)

    @classmethod
    def from_settings(cls, settings: Any) -> SafetyConstraints:
        """Build from ``SafetySettings`` or another object with ``model_dump``."""

        if isinstance(settings, Mapping):
            return cls.from_mapping(settings)
        model_dump = getattr(settings, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return cls.from_mapping(dumped)
        raise TypeError("settings must be a mapping or expose model_dump()")

    @property
    def safety_case_hash(self) -> str:
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        """Method-form alias useful to configuration and evidence callers."""

        return self.safety_case_hash


def safety_case_hash(constraints: SafetyConstraints) -> str:
    """Return the stable hash for a set of enforced constraints."""

    return constraints.safety_case_hash


DEFAULT_SAFETY_CONSTRAINTS = SafetyConstraints()
