"""MINLP search for binary classifiers.

An older, simpler pipeline than the regression search: for each categorical
combination, SLSQP maximises the priority benefit subject to the real
model's probability being on the requested side of the threshold. See known
issues MINLP-4 to MINLP-6.
"""

import traceback

import numpy as np
from scipy.optimize import minimize

from explainit.explainers.minlp.intervals import (
    allowed_interval_constraint_value,
    project_to_allowed_intervals,
)
from explainit.logging_config import logger


def optimise_for_class(
    start,
    numerical_indices,
    numerical_priorities,
    bounds,
    model,
    objective,
    target_class,
    threshold,
    max_iterations,
):
    """Run SLSQP from ``start`` with the classification constraint.

    Returns:
        tuple | None: ``(counterfactual, prediction, score)`` if the result is
        on the requested side of the threshold, otherwise ``None``.
    """

    def to_full(x, start=start):
        full = start.copy()
        for i, idx in enumerate(numerical_indices):
            if i < len(x):
                full[idx] = x[i]
        return full

    def benefit(x):
        try:
            return objective(to_full(x))
        except ValueError:
            return -1e12

    def classification_constraint(x):
        pred = model([to_full(x)])[0]
        if target_class == 1:
            return pred - threshold
        return threshold - pred

    constraints = [{"type": "ineq", "fun": classification_constraint}]

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

    logger.debug(f"Initial values: {x0}")
    logger.debug(f"Bounds: {bounds_list}")

    result = minimize(
        lambda x: -benefit(x),
        x0,
        method="SLSQP",
        bounds=bounds_list,
        constraints=constraints,
        options={"maxiter": max_iterations, "ftol": 1e-6},
    )

    # Partial convergence is accepted; validity is checked with the model.
    solution = start.copy()
    for i, idx in enumerate(numerical_indices):
        solution[idx] = result.x[i]
    pred = model([solution])[0]
    is_valid = (pred > threshold) if target_class == 1 else (pred < threshold)
    logger.debug(f"Solution prediction: {pred:.4f}, valid: {is_valid}, message: {result.message}")
    if not is_valid:
        logger.debug(
            f"Solution doesn't meet classification constraint: pred={pred:.4f}, threshold={threshold}"
        )
        return None
    return solution, pred, benefit(result.x)


def search_combinations(
    warm_starts,
    sample,
    numerical_priorities,
    bounds,
    model,
    objective,
    target_class,
    threshold,
    max_iterations,
):
    """:func:`optimise_for_class` for every warm start.

    Returns:
        tuple: ``(counterfactuals, predictions, scores, combination_numbers)``.
    """
    counterfactuals, predictions, scores, found_counts = [], [], [], []
    numerical_indices = list(numerical_priorities.keys())

    for combo_id, values in warm_starts.items():
        logger.info(f"Processing categorical combination {combo_id}...")
        start = sample.copy()
        if values.get("initial_solution"):
            initial_numerical = values["initial_solution"]
            categorical = values.get("categorical_combo", {})
        else:
            initial_numerical, categorical = {}, {}
        for idx, val in categorical.items():
            start[idx] = val
        for idx, val in initial_numerical.items():
            start[idx] = val

        logger.info(f"Starting optimization for combination {combo_id}...")
        try:
            found = optimise_for_class(
                start,
                numerical_indices,
                numerical_priorities,
                bounds,
                model,
                objective,
                target_class,
                threshold,
                max_iterations,
            )
        except Exception as e:
            logger.warning(f"Error during optimization for combination {combo_id}: {e}")
            logger.debug(traceback.format_exc())
            continue
        if found is None:
            continue
        solution, pred, score = found
        counterfactuals.append(solution)
        predictions.append(pred)
        scores.append(score)
        found_counts.append(combo_id + 1)
        logger.info(f"Found valid counterfactual: pred={pred:.4f}, weight={scores[-1]:.4f}")

    return counterfactuals, predictions, scores, found_counts


def rank(counterfactuals, predictions, scores, found_counts, return_top_n=None):
    """Sort the results by ascending score (known issue MINLP-5) and keep ``return_top_n``."""
    if not counterfactuals:
        logger.warning("No valid counterfactuals found!")
        return counterfactuals, predictions, scores, found_counts

    order = np.argsort(scores)
    counterfactuals = [counterfactuals[i] for i in order]
    predictions = [predictions[i] for i in order]
    scores = [scores[i] for i in order]
    found_counts = [found_counts[i] for i in order]

    if return_top_n is not None and len(counterfactuals) > return_top_n:
        counterfactuals = counterfactuals[:return_top_n]
        predictions = predictions[:return_top_n]
        scores = scores[:return_top_n]
        found_counts = found_counts[:return_top_n]

    logger.info(f"Found {len(counterfactuals)} valid counterfactuals")
    return counterfactuals, predictions, scores, found_counts
