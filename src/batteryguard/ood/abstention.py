"""Deterministic abstention policy for personalized predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    abstain: bool
    reasons: tuple[str, ...]
    interval_width_ratio: float


class AbstentionPolicy:
    def __init__(
        self,
        *,
        quality_threshold: float = 0.90,
        ood_threshold: float = 0.98,
        interval_width_ratio_max: float = 0.45,
        required_early_cycles: int = 30,
    ) -> None:
        if not 0 <= quality_threshold <= 1:
            raise ValueError("quality_threshold must be in [0, 1]")
        if not 0 < ood_threshold <= 1:
            raise ValueError("ood_threshold must be in (0, 1]")
        if interval_width_ratio_max <= 0 or required_early_cycles < 1:
            raise ValueError("invalid interval/cycle abstention threshold")
        self.quality_threshold = float(quality_threshold)
        self.ood_threshold = float(ood_threshold)
        self.interval_width_ratio_max = float(interval_width_ratio_max)
        self.required_early_cycles = int(required_early_cycles)

    def evaluate(
        self,
        *,
        quality_score: float | None,
        ood_score: float | None,
        point_estimate: float | None,
        interval_low: float | None,
        interval_high: float | None,
        observed_cycles: int,
        protocol_available: bool,
        model_version: str | None = None,
        evidence_model_version: str | None = None,
    ) -> AbstentionDecision:
        reasons: list[str] = []
        if quality_score is None or not np.isfinite(quality_score):
            reasons.append("MISSING_DATA_QUALITY")
        elif quality_score < self.quality_threshold:
            reasons.append("LOW_DATA_QUALITY")
        if ood_score is None or not np.isfinite(ood_score):
            reasons.append("MISSING_OOD_SCORE")
        elif ood_score >= self.ood_threshold:
            reasons.append("OUT_OF_DISTRIBUTION")

        values = (point_estimate, interval_low, interval_high)
        if any(value is None or not np.isfinite(value) for value in values):
            ratio = float("inf")
            reasons.append("MISSING_OR_INVALID_INTERVAL")
        else:
            point = float(point_estimate)  # type: ignore[arg-type]
            low = float(interval_low)  # type: ignore[arg-type]
            high = float(interval_high)  # type: ignore[arg-type]
            if point <= 0 or low > point or point > high:
                ratio = float("inf")
                reasons.append("MALFORMED_INTERVAL")
            else:
                ratio = (high - low) / point
                if ratio > self.interval_width_ratio_max:
                    reasons.append("INTERVAL_TOO_WIDE")
        if observed_cycles < self.required_early_cycles:
            reasons.append("INSUFFICIENT_EARLY_CYCLES")
        if not protocol_available:
            reasons.append("MISSING_PROTOCOL_METADATA")
        if (
            model_version is not None
            and evidence_model_version is not None
            and model_version != evidence_model_version
        ):
            reasons.append("MODEL_EVIDENCE_VERSION_MISMATCH")
        unique_reasons = tuple(dict.fromkeys(reasons))
        return AbstentionDecision(bool(unique_reasons), unique_reasons, float(ratio))


def should_abstain(**kwargs: object) -> AbstentionDecision:
    """Convenience wrapper using the default policy thresholds."""

    return AbstentionPolicy().evaluate(**kwargs)  # type: ignore[arg-type]
