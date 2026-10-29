"""Stage 4.5: warm starts for the optimiser.

For every categorical combination, a linear program finds numerical values for
which the surrogate hits the pass target. These points start the nonlinear
optimisation of Stage 5.
"""

import numpy as np

from explainit.explainers.minlp.linear_program import solve_linear_constraint
from explainit.logging_config import logger


def solve_warm_starts(
    targets,
    combinations,
    indices,
    coefficients,
    bounds,
    sample,
    surrogate,
    pass_target,
    model,
    target,
    epsilon,
):
    """One linear-program warm start per categorical combination.

    Args:
        targets: ``{combination_id: right-hand side}`` of ``coefficients . x``.
        combinations: ``{combination_id: {column: value}}``.
        indices: Columns of the mutable numerical features, in LP order.
        coefficients: LP coefficients, aligned with ``indices``.
        bounds: ``(lo, hi)`` per LP variable, aligned with ``indices``.
        sample: Current point; other columns keep its values.
        surrogate: ``h(x)`` for a full feature vector.
        pass_target: ``pass_target(combination_id)`` the surrogate aims at.
        model: The real model, evaluated once per warm start for diagnostics.
        target: The real target.
        epsilon: Tolerance of the target band.

    Returns:
        tuple: ``(warm_starts, gaps)``. ``warm_starts`` maps a combination id to
        ``{"initial_solution": {column: value}, "categorical_combo": {...}}``;
        ``gaps`` lists ``(|f(x0) - target|, |h(x0) - pass target|)`` per warm start.
    """
    logger.info(
        "[Stage 4.5] Solving an LP per categorical combination "
        "(coefficients . x = adjusted_target, bounded). The first feasible "
        "x becomes the warm-start for SLSQP later on."
    )
    warm_starts = {}
    gaps = []

    for combination_id, lp_target in targets.items():
        logger.info(
            "[Stage 4.5] Combo %d/%d -> LP target=%.4f",
            combination_id + 1,
            len(targets),
            float(lp_target),
        )
        combination = combinations[combination_id]
        logger.debug("[Stage 4.5] Combo %d categorical assignment: %s", combination_id, combination)
        x0 = sample.copy()
        for idx, val in combination.items():
            x0[idx] = val

        linear_solution = solve_linear_constraint(
            coefficients, lp_target, bounds, method="auto", tolerance=epsilon
        )
        logger.info(
            "[Stage 4.5] Combo %d LP %s (method=%s).",
            combination_id,
            "feasible" if linear_solution["success"] else "infeasible",
            linear_solution.get("method_used"),
        )
        solution = linear_solution["solution"]
        logger.debug("[Stage 4.5] Combo %d LP solution=%s", combination_id, solution)
        if solution is None:
            continue

        initial_solution = dict(zip(indices, solution, strict=False))
        warm_starts[combination_id] = {
            "initial_solution": initial_solution,
            "categorical_combo": combination,
        }
        for key, value in initial_solution.items():
            x0[key] = value
        logger.debug("[Stage 4.5] Combo %d candidate vector: %s", combination_id, x0)

        check_value = surrogate(x0)
        aim = pass_target(combination_id)
        assert abs(check_value - aim) <= epsilon, (
            f"Constraint function value {check_value} exceeds tolerance {epsilon} from target {aim}"
        )
        linear_gap = abs(float(check_value) - float(aim))
        logger.info(
            "[Stage 4.5] Combo %d sanity check: linearised h(x)=%.4f "
            "(target=%.4f, |gap|=%.4f, epsilon=%.4f).",
            combination_id,
            float(check_value),
            float(aim),
            linear_gap,
            float(epsilon),
        )
        model_prediction = float(np.asarray(model([x0])).reshape(-1)[0])
        model_gap = abs(model_prediction - float(target))
        gaps.append((model_gap, linear_gap))
        logger.info(
            "[Stage 4.5] Combo %d warm-start true model_pred=%.4f "
            "(|model gap|=%.4f vs linear gap=%.4f).",
            combination_id,
            model_prediction,
            model_gap,
            linear_gap,
        )

    logger.info(
        "[Stage 4.6] %d/%d categorical combination(s) yielded an initial feasible solution.",
        len(warm_starts),
        len(targets),
    )
    return warm_starts, gaps


def warm_start_health(targets, warm_starts, gaps):
    """Summary of the first pass's warm starts, reported in the search result."""
    best = min(gaps, key=lambda t: t[0]) if gaps else (None, None)
    return {
        "total_combos": len(targets),
        "feasible_combos": len(warm_starts),
        "best_warmstart_model_gap": (float(best[0]) if best[0] is not None else None),
        "best_warmstart_linear_gap": (float(best[1]) if best[1] is not None else None),
    }
