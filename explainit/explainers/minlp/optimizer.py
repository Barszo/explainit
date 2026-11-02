"""Stage 5: maximise the priority benefit subject to the surrogate constraint.

For one categorical combination, SLSQP optimises the numerical features so
that the surrogate ``h(x)`` stays within ``epsilon`` of the pass target, the
features stay in their (trust-region) bounds and allowed intervals, and the
priority benefit is as high as possible.
"""

import numpy as np
from scipy.optimize import minimize

from explainit.explainers.minlp.intervals import (
    allowed_interval_constraint_value,
    project_to_allowed_intervals,
)
from explainit.logging_config import logger


def optimise_combination(
    start,
    numerical_indices,
    numerical_priorities,
    bounds,
    surrogate,
    objective,
    pass_target,
    epsilon,
    label="",
):
    """Run SLSQP from ``start`` over the numerical features.

    Args:
        start: Full feature vector (warm start with its categorical assignment).
            Columns that are not optimised keep these values.
        numerical_indices: Columns optimised by SLSQP, in order.
        numerical_priorities: Per-feature configs (allowed intervals).
        bounds: ``{column: (lo, hi)}``.
        surrogate: ``h(x)`` for a full feature vector.
        objective: Priority benefit of a full feature vector; may raise
            ``ValueError`` for disallowed values.
        pass_target: Value ``h`` must be within ``epsilon`` of.
        epsilon: Half-width of the target band.
        label: Prefix for log messages.

    Returns:
        tuple: ``(result, solution, h_solution)`` -- the SciPy result, the full
        feature vector and the surrogate value there.
    """

    def to_full(x, start=start):
        full = start.copy()
        for j, idx in enumerate(numerical_indices):
            full[idx] = x[j]
        return full

    def h(x):
        return surrogate(to_full(x))

    def negative_benefit(x):
        try:
            benefit = objective(to_full(x))
        except ValueError:
            benefit = -1e12
        return -benefit

    def lower_constraint(x):
        return h(x) - (pass_target - epsilon)

    def upper_constraint(x):
        return (pass_target + epsilon) - h(x)

    constraints = [
        {"type": "ineq", "fun": lower_constraint},
        {"type": "ineq", "fun": upper_constraint},
    ]
    x0 = np.array([start[idx] for idx in numerical_indices])
    bounds_list = [bounds[idx] for idx in numerical_indices]
    for j, feature_idx in enumerate(numerical_indices):
        cfg = numerical_priorities.get(feature_idx, {})
        allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None
        if allowed_intervals:
            x0[j] = project_to_allowed_intervals(float(x0[j]), allowed_intervals)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": (
                        lambda x, _j=j, _intervals=allowed_intervals: (
                            allowed_interval_constraint_value(float(x[_j]), _intervals)
                        )
                    ),
                }
            )
    for j, (lo, hi) in enumerate(bounds_list):
        if lo is not None:
            x0[j] = max(float(x0[j]), float(lo))
        if hi is not None:
            x0[j] = min(float(x0[j]), float(hi))
    logger.info("%s Initial numerical x0=%s", label, x0)
    logger.debug(
        "%s Constraints (bounded): h(x) in [%.4f, %.4f]",
        label,
        pass_target - epsilon,
        pass_target + epsilon,
    )
    logger.debug("%s Bounds list passed to SLSQP: %s", label, bounds_list)
    logger.info(
        "%s Running SLSQP (maximise priority weight subject to Shapley-linear constraint)...",
        label,
    )
    result = minimize(
        negative_benefit,
        x0,
        method="SLSQP",
        bounds=bounds_list,
        constraints=constraints,
    )
    return result, to_full(result.x), h(result.x)
