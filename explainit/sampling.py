"""Sampling feature values in proportion to their priorities.

Used by the random search and by the probability-distribution plots, so both
always describe the same sampling behaviour.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


def sample_numeric_value(
    constraint: Mapping[str, Any], max_tries: int = 100, rng: Any = None
) -> float:
    """Draw a value from ``[min, max]`` with density proportional to the priority.

    Uses rejection sampling against ``constraint["function"]`` (values in
    ``[0, 1]``); if no sample is accepted within ``max_tries`` attempts, falls
    back to inverse-transform sampling on a 256-point grid.

    Args:
        constraint: ``{"min": ..., "max": ..., "function": ...}``.
        max_tries: Rejection-sampling attempts before the fallback.
        rng: ``numpy.random.RandomState`` (default: the global ``numpy.random``).
    """
    if rng is None:
        rng = np.random

    min_val = float(constraint["min"])
    max_val = float(constraint["max"])
    f = constraint["function"]

    for _ in range(max_tries):
        rv = rng.uniform(min_val, max_val)
        w = float(np.asarray(f(rv)).squeeze())
        if rng.random() < w:
            return float(rv)

    xs = np.linspace(min_val, max_val, 256)
    ws = np.asarray(f(xs)).astype(float).ravel()
    ws = np.clip(ws, 0.0, None)
    if ws.sum() == 0 or not np.isfinite(ws).all():
        return float(rng.uniform(min_val, max_val))
    p = ws / ws.sum()
    idx = rng.choice(len(xs), p=p)
    jitter = (max_val - min_val) / 256 * (rng.random() - 0.5)
    return float(np.clip(xs[idx] + jitter, min_val, max_val))


def sample_category(group, weights: Mapping, weighted: bool, np_rng, py_rng) -> tuple:
    """Pick a categorical combination; combinations with weight ``<= 0`` or ``None`` are never picked.

    Args:
        group: Column indices of the group (for the error message).
        weights: ``{combination: weight}``.
        weighted: Pick in proportion to the weights; otherwise uniformly.
        np_rng: Generator for weighted picks.
        py_rng: ``random.Random``-like generator for uniform picks.

    Raises:
        ValueError: If every combination is forbidden.
    """
    combos = list(weights.keys())
    values = np.array(list(weights.values()), dtype=float)
    allowed_mask = values > 0
    if not np.any(allowed_mask):
        raise ValueError(
            f"All combinations for categorical group {group} are forbidden (weight = 0). At least one combination must have weight > 0."
        )
    allowed = [combos[j] for j in range(len(combos)) if allowed_mask[j]]
    if weighted:
        allowed_weights = values[allowed_mask]
        allowed_weights = allowed_weights / allowed_weights.sum()
        return allowed[np_rng.choice(len(allowed), p=allowed_weights)]
    return py_rng.choice(allowed)
