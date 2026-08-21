"""Safety-gated multi-objective charging-policy search."""

from batteryguard.optimizer.objectives import ObjectiveVector, dominates, objective_vector
from batteryguard.optimizer.pareto import (
    compute_pareto_front,
    is_safety_feasible,
    mark_pareto_optimal,
    pareto_front,
    pareto_policy_ids,
    select_pareto,
)
from batteryguard.optimizer.policy_space import (
    PolicySpace,
    balanced_policy,
    build_policy_library,
    default_policies,
    fast_policy,
    get_policy_library,
    life_policy,
    policy_library,
)
from batteryguard.optimizer.search import (
    PolicyOptimizer,
    PolicySearch,
    SearchContext,
    evaluate_candidates,
)

__all__ = [
    "ObjectiveVector",
    "PolicyOptimizer",
    "PolicySearch",
    "PolicySpace",
    "SearchContext",
    "balanced_policy",
    "build_policy_library",
    "compute_pareto_front",
    "default_policies",
    "dominates",
    "evaluate_candidates",
    "fast_policy",
    "get_policy_library",
    "is_safety_feasible",
    "life_policy",
    "mark_pareto_optimal",
    "objective_vector",
    "pareto_front",
    "pareto_policy_ids",
    "policy_library",
    "select_pareto",
]
