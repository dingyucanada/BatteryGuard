"""Transparent B0/B1/B2 lifetime-prediction ladder."""

from batteryguard.prediction.baselines import (
    ElasticNetLifetimeModel,
    LinearLifetimeModel,
    MedianLifetimeModel,
    ModelNotFittedError,
    RidgeLifetimeModel,
)
from batteryguard.prediction.evaluation import (
    EvaluationReport,
    GroupedEvaluationResult,
    RegressionMetrics,
    evaluate_regression,
    grouped_evaluate,
)
from batteryguard.prediction.registry import ModelLadder, build_model
from batteryguard.prediction.xgboost_model import XGBoostLifetimeModel

__all__ = [
    "ElasticNetLifetimeModel",
    "EvaluationReport",
    "GroupedEvaluationResult",
    "LinearLifetimeModel",
    "MedianLifetimeModel",
    "ModelLadder",
    "ModelNotFittedError",
    "RegressionMetrics",
    "RidgeLifetimeModel",
    "XGBoostLifetimeModel",
    "build_model",
    "evaluate_regression",
    "grouped_evaluate",
]
