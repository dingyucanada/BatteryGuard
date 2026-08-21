"""Out-of-distribution scoring and conservative abstention."""

from batteryguard.ood.abstention import AbstentionDecision, AbstentionPolicy, should_abstain
from batteryguard.ood.mahalanobis import MahalanobisNotFittedError, MahalanobisOOD

__all__ = [
    "AbstentionDecision",
    "AbstentionPolicy",
    "MahalanobisNotFittedError",
    "MahalanobisOOD",
    "should_abstain",
]
