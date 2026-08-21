"""Conservative fixed policy used when a candidate cannot be authorized."""

from __future__ import annotations

from batteryguard.schemas.policy import ChargingPolicy, PolicyFamily

FALLBACK_POLICY_ID = "conservative-cccv-v1"


def conservative_fallback_policy(target_soc: float = 0.80) -> ChargingPolicy:
    """Build a low-rate, non-personalized fixed policy.

    The policy still requires simulation and SafetyShield evaluation.  Being a
    fallback never grants implicit authorization.
    """

    if not 0 < target_soc <= 1:
        raise ValueError("target_soc must lie in (0, 1]")
    if target_soc <= 0.65:
        breaks = [target_soc]
        rates = [0.45]
    else:
        breaks = [0.65, target_soc]
        rates = [0.55, 0.35]
    return ChargingPolicy(
        policy_id=FALLBACK_POLICY_ID,
        family=PolicyFamily.FALLBACK,
        c_rates=rates,
        soc_breaks=breaks,
        v_max=3.45,
        cv_cutoff_c=0.05,
        target_soc=target_soc,
        personalized=False,
    )


fallback_policy = conservative_fallback_policy
get_fallback_policy = conservative_fallback_policy
