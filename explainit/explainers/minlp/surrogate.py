"""The Shapley-linear surrogate model used by the optimiser.

Around the current point ``x^k`` the model is approximated by

    h(x) = f(x^k) + sum_i c_i (x_i - x^k_i) + sum_g phi_g [x_g == anchor_g]

where ``c_i = phi_i / (anchor_i - x^k_i)`` is the per-unit Shapley value of
numerical feature ``i`` and ``phi_g`` the Shapley value of categorical group
``g`` (counted when the group takes the anchor's category).
"""

import numpy as np


def unit_coefficients(numerical_shapley, sample, anchor, numerical_indices, immutable):
    """Per-unit coefficients ``c_i`` and ``sum_i c_i x^k_i`` over mutable features.

    A feature with the same value in ``sample`` and ``anchor`` gets
    coefficient 0 (its unit effect is undefined).

    Returns:
        tuple: ``(coefficients, coefficient_times_sample)``.
    """
    coefficients = {}
    coefficient_times_sample = 0.0
    for key in numerical_indices:
        denom = anchor[key] - sample[key]
        coefficient = 0.0 if abs(float(denom)) < 1e-12 else numerical_shapley[key] / denom
        coefficients[key] = coefficient
        if key not in immutable:
            coefficient_times_sample += coefficient * sample[key]
    return coefficients, coefficient_times_sample


def _categorical_term(values, group, group_shapley, sample, anchor):
    current = tuple(float(values[idx]) for idx in group)
    if current == tuple(float(sample[idx]) for idx in group):
        return 0.0
    if current == tuple(float(anchor[idx]) for idx in group):
        return group_shapley
    raise ValueError(f"Invalid categorical combination encountered for indices {group}: {current}")


def categorical_contribution(combination, groups, categorical_shapley, sample, anchor):
    """Surrogate contribution of a categorical assignment ``{column: value}``."""
    values = np.zeros(max(combination.keys()) + 1, dtype=float)
    for idx, val in combination.items():
        values[idx] = val
    result = 0.0
    for group in groups:
        result += _categorical_term(values, group, categorical_shapley[group], sample, anchor)
    return result


def surrogate_prediction(
    x, coefficients, shapley, sample, anchor, priorities_for_search, basic_prediction
):
    """``h(x)`` for a full-length feature vector ``x``.

    Args:
        x: Candidate feature vector.
        coefficients: Per-unit numerical coefficients (:func:`unit_coefficients`).
        shapley: ``{"numerical": ..., "categorical": {group: phi}}``.
        sample: The point the surrogate is linearised at.
        anchor: The anchor exemplar.
        priorities_for_search: Priorities whose features enter ``h``.
        basic_prediction: ``f(sample)``.

    Raises:
        ValueError: If a categorical group is neither the sample's nor the
            anchor's category.
    """
    result = basic_prediction
    for key in priorities_for_search["numerical"]:
        result += coefficients[key] * (x[key] - sample[key])
    for group in priorities_for_search["categorical"]:
        result += _categorical_term(x, group, shapley["categorical"][group], sample, anchor)
    return result


def reachable_range(basic_prediction, coefficients, bounds, sample, immutable) -> tuple:
    """Interval ``h`` can span inside ``bounds`` (interval arithmetic, no model calls)."""
    lo_total = hi_total = float(basic_prediction)
    for idx, coefficient in (coefficients or {}).items():
        if idx in immutable:
            continue
        lo, hi = bounds.get(idx, (None, None))
        if lo is None or hi is None:
            continue
        x = float(sample[idx])
        d_lo = float(coefficient) * (float(lo) - x)
        d_hi = float(coefficient) * (float(hi) - x)
        lo_total += min(d_lo, d_hi)
        hi_total += max(d_lo, d_hi)
    return lo_total, hi_total
