"""Transparent fixed strategy families for the MVP policy search."""

from __future__ import annotations

from dataclasses import dataclass

from batteryguard.safety.fallback import conservative_fallback_policy
from batteryguard.schemas.policy import ChargingPolicy, PolicyFamily


def fast_policy(target_soc: float = 0.80) -> ChargingPolicy:
    return _scaled_policy(
        policy_id="fast-mscc-v1",
        family=PolicyFamily.FAST,
        rates=[2.50, 2.00, 1.40, 0.80],
        fractions=[0.3125, 0.625, 0.90, 1.00],
        target_soc=target_soc,
        v_max=3.60,
        cv_cutoff_c=0.08,
    )


def balanced_policy(target_soc: float = 0.80) -> ChargingPolicy:
    return _scaled_policy(
        policy_id="balanced-mscc-v1",
        family=PolicyFamily.BALANCED,
        rates=[1.80, 1.45, 1.00, 0.60],
        fractions=[0.3125, 0.625, 0.90, 1.00],
        target_soc=target_soc,
        v_max=3.55,
        cv_cutoff_c=0.07,
    )


def life_policy(target_soc: float = 0.80) -> ChargingPolicy:
    return _scaled_policy(
        policy_id="life-mscc-v1",
        family=PolicyFamily.LIFE,
        rates=[1.00, 0.80, 0.60, 0.40],
        fractions=[0.3125, 0.625, 0.90, 1.00],
        target_soc=target_soc,
        v_max=3.50,
        cv_cutoff_c=0.05,
    )


def _scaled_policy(
    *,
    policy_id: str,
    family: PolicyFamily,
    rates: list[float],
    fractions: list[float],
    target_soc: float,
    v_max: float,
    cv_cutoff_c: float,
) -> ChargingPolicy:
    if not 0 < target_soc <= 1:
        raise ValueError("target_soc must lie in (0, 1]")
    breaks = [round(target_soc * fraction, 10) for fraction in fractions]
    breaks[-1] = target_soc
    return ChargingPolicy(
        policy_id=policy_id,
        family=family,
        c_rates=rates,
        soc_breaks=breaks,
        v_max=v_max,
        cv_cutoff_c=cv_cutoff_c,
        target_soc=target_soc,
        personalized=False,
    )


def get_policy_library(target_soc: float = 0.80) -> dict[PolicyFamily, ChargingPolicy]:
    """Return fresh FAST/BALANCED/LIFE/FALLBACK policy objects."""

    policies = (
        fast_policy(target_soc),
        balanced_policy(target_soc),
        life_policy(target_soc),
        conservative_fallback_policy(target_soc),
    )
    return {policy.family: policy for policy in policies}


def default_policies(
    target_soc: float = 0.80,
    *,
    include_fallback: bool = False,
) -> list[ChargingPolicy]:
    """Return the three normal candidates, optionally adding the fallback."""

    library = get_policy_library(target_soc)
    ordered_families = [PolicyFamily.FAST, PolicyFamily.BALANCED, PolicyFamily.LIFE]
    if include_fallback:
        ordered_families.append(PolicyFamily.FALLBACK)
    return [library[family] for family in ordered_families]


@dataclass(frozen=True, slots=True)
class PolicySpace:
    """Small deterministic strategy library used by the MVP optimizer."""

    target_soc: float = 0.80
    include_fallback: bool = False

    def library(self) -> dict[PolicyFamily, ChargingPolicy]:
        return get_policy_library(self.target_soc)

    def candidates(self) -> list[ChargingPolicy]:
        return default_policies(
            self.target_soc,
            include_fallback=self.include_fallback,
        )

    def by_family(self, family: PolicyFamily | str) -> ChargingPolicy:
        return self.library()[PolicyFamily(family)]


policy_library = get_policy_library
build_policy_library = get_policy_library
