"""Stage 0 bounds and the trust region.

Stage 0 derives each numerical feature's allowed region from its priority
function. The trust region limits how far one optimisation pass may move each
feature, because the Shapley-linear surrogate is only accurate near the point
it was linearised at.
"""

from dataclasses import dataclass

import numpy as np

from explainit.explainers.minlp.intervals import extract_positive_intervals
from explainit.logging_config import logger


def derive_bounds_and_intervals(
    numerical_priorities, dataset, sample, feature_label, grid_size=2000
):
    """Write ``min``, ``max`` and ``allowed_intervals`` into each feature config.

    The allowed region is where the priority function is positive within the
    dataset's range. Immutable features are fixed at their configured value.
    The configs are modified in place; :class:`MINLPSearchExplainer` passes a
    copy for every search (see known issue MINLP-1).

    Raises:
        ValueError: If a feature has no positive-priority region.
    """
    logger.info("--- STAGE 0/6: Derive numerical bounds from priority functions ---")
    logger.info(
        "For each numerical feature, infer feasible ranges from where "
        "priority function > 0 over dataset support; this overrides user "
        "min/max and excludes internal zero-priority gaps."
    )

    for idx, cfg in numerical_priorities.items():
        if not isinstance(cfg, dict):
            continue
        fn = cfg.get("function")
        dmin = float(np.min(dataset[:, idx]))
        dmax = float(np.max(dataset[:, idx]))

        if fn is None:
            fixed_val = float(cfg.get("min", sample[idx]))
            cfg["min"] = fixed_val
            cfg["max"] = fixed_val
            cfg["allowed_intervals"] = [(fixed_val, fixed_val)]
            continue

        if abs(dmax - dmin) < 1e-12:
            cfg["min"] = dmin
            cfg["max"] = dmax
            cfg["allowed_intervals"] = [(dmin, dmax)]
            continue

        x_grid = np.linspace(dmin, dmax, int(grid_size))
        y_values = []
        for x in x_grid:
            try:
                y = float(fn(float(x)))
            except Exception:
                y = 0.0
            if not np.isfinite(y):
                y = 0.0
            y_values.append(y)
        y_grid = np.array(y_values, dtype=float)
        y_grid[~np.isfinite(y_grid)] = 0.0

        intervals = extract_positive_intervals(x_grid, y_grid, positive_eps=1e-12)
        if not intervals:
            raise ValueError(
                f"Feature {idx} has no positive-priority region on dataset range "
                f"[{dmin}, {dmax}]. Cannot derive feasible bounds."
            )

        derived_min = float(intervals[0][0])
        derived_max = float(intervals[-1][1])
        cfg["min"] = max(dmin, derived_min)
        cfg["max"] = min(dmax, derived_max)
        cfg["allowed_intervals"] = intervals

        logger.info(
            "[Stage 0] Feature %s (idx=%d): derived bounds=[%.6f, %.6f] "
            "| dataset=[%.6f, %.6f] | allowed_intervals=%s",
            feature_label(idx),
            idx,
            float(cfg["min"]),
            float(cfg["max"]),
            dmin,
            dmax,
            intervals,
        )


@dataclass
class TrustRegion:
    """Step limit as a fraction of each feature's allowed span.

    ``fraction=None`` disables the trust region. The radius grows after an
    accepted step and shrinks after a rejected one, within ``[minimum, maximum]``.
    """

    fraction: float | None = None
    minimum: float = 0.02
    maximum: float = 1.0
    expand: float = 2.0
    contract: float = 0.5

    @property
    def enabled(self) -> bool:
        return self.fraction is not None

    def grow(self) -> bool:
        """Enlarge the region; ``False`` when disabled or already at full span."""
        if self.fraction is None or self.fraction >= self.maximum:
            return False
        self.fraction = min(self.maximum, self.fraction * self.expand)
        logger.info(
            "[Trust region] Enlarged to %.3f of each feature's allowed span.", self.fraction
        )
        return True

    def shrink(self) -> None:
        if self.fraction is None:
            return
        self.fraction = max(self.minimum, self.fraction * self.contract)
        logger.info("[Trust region] Shrunk to %.3f of each feature's allowed span.", self.fraction)

    def limit(self, bounds, center) -> dict[int, tuple]:
        """Per-feature ``(lo, hi)``: ``bounds`` intersected with the region around ``center``."""
        if self.fraction is None:
            return dict(bounds)
        center = np.asarray(center, dtype=float)
        limited = {}
        for idx, (lo, hi) in bounds.items():
            if lo is None or hi is None:
                limited[idx] = (lo, hi)
                continue
            lo, hi = float(lo), float(hi)
            # The iterate can sit outside its own allowed range (the priority
            # support need not contain the sample), so clamp before expanding.
            c = min(max(float(center[idx]), lo), hi)
            delta = float(self.fraction) * (hi - lo)
            limited[idx] = (max(lo, c - delta), min(hi, c + delta))
        return limited
