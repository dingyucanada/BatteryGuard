"""Leakage-safe early-cycle feature engineering."""

from batteryguard.features.delta_q import compute_delta_q_features, delta_q_curve
from batteryguard.features.pipeline import EarlyCycleFeaturePipeline, FeatureExtractionError
from batteryguard.features.protocol import (
    derive_protocol_features,
    parse_c_rates,
    protocol_stage_features,
)
from batteryguard.features.summary import summarize_early_cycles
from batteryguard.features.trends import linear_trend, quadratic_curvature

__all__ = [
    "EarlyCycleFeaturePipeline",
    "FeatureExtractionError",
    "compute_delta_q_features",
    "delta_q_curve",
    "derive_protocol_features",
    "linear_trend",
    "parse_c_rates",
    "protocol_stage_features",
    "quadratic_curvature",
    "summarize_early_cycles",
]
