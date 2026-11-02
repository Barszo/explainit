"""Allowed-interval helpers for numerical features.

A numerical feature may be restricted to one or more disjoint intervals
(``[(lo, hi), ...]``), derived in Stage 0 from where its priority is positive.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from explainit.explainers.minlp._types import Interval
from explainit.typing import FloatArray


def in_allowed_intervals(value: float, intervals: Sequence[Interval], tol: float = 1e-12) -> bool:
    """Whether ``value`` lies in any interval (with tolerance ``tol``)."""
    if not intervals:
        return False
    return any(value >= lo - tol and value <= hi + tol for lo, hi in intervals)


def allowed_interval_constraint_value(value: float, intervals: Sequence[Interval]) -> float:
    """SLSQP inequality: non-negative exactly when ``value`` is inside an interval."""
    if not intervals:
        return -1.0
    return float(max(min(value - lo, hi - value) for lo, hi in intervals))


def project_to_allowed_intervals(value: float, intervals: Sequence[Interval]) -> float:
    """``value`` itself if allowed, otherwise the nearest interval end point."""
    if not intervals:
        return value
    if in_allowed_intervals(value, intervals):
        return value
    candidates = []
    for lo, hi in intervals:
        candidates.append(lo)
        candidates.append(hi)
    return float(min(candidates, key=lambda c: abs(c - value)))


def extract_positive_intervals(
    x_grid: FloatArray, y_grid: npt.ArrayLike, positive_eps: float = 1e-12
) -> list[Interval]:
    """Maximal runs of grid points where ``y_grid > positive_eps``, as intervals."""
    mask = np.asarray(y_grid, dtype=float) > float(positive_eps)
    intervals = []
    i = 0
    n = len(x_grid)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        intervals.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return intervals


def clip_to_allowed(
    value: float, config: object, bounds: tuple[float | None, float | None]
) -> float:
    """Project ``value`` onto a feature's allowed intervals, then clip to ``bounds``."""
    intervals = config.get("allowed_intervals") if isinstance(config, dict) else None
    v = float(value)
    if intervals:
        v = project_to_allowed_intervals(v, intervals)
    lo, hi = bounds
    if lo is not None:
        v = max(v, float(lo))
    if hi is not None:
        v = min(v, float(hi))
    return v
