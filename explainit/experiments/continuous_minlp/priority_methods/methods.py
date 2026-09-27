"""Priority-branch method registry: MINLP search + random-search baseline.

Both methods consume a *materialised* priority dict (the output of
``build_priorities`` for a specific sample) and return one or more
counterfactual vectors. Every method exposes::

    method = build_method(name, pctx=..., epsilon=...)
    out = method.generate_many(x, target, priorities, n_cfs)
    # out = {"cfs": [np.ndarray, ...], "cf_iterations": [int, ...] | None,
    #        "iterations": int | None, "error": str | None}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.explainers.minlp_search import MINLSearchExplainer  # noqa: E402
from explainit.explainers.random_search import RandomSearchExplainer  # noqa: E402

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.priority_methods.methods"
)

workflow_logger = logging.getLogger("explainit.workflow")


def compute_priority_score(priorities: Dict[str, Any], cf: np.ndarray) -> float:
    """Overall priority score of a counterfactual.

    Mirrors ``RandomSearchExplainer.calculate_preference_score``: the sum of
    the per-feature priority weights (numerical function values for actionable
    features, plus the categorical weight of the active category). ``None``
    weights (forbidden/pinned categories) contribute 0.
    """
    cf = np.asarray(cf, dtype=float)
    scores: List[float] = []
    for idx, constraint in priorities.get("numerical", {}).items():
        if isinstance(constraint, dict) and constraint.get("function") is not None:
            scores.append(float(np.asarray(constraint["function"](cf[idx])).squeeze()))
    for group_indices, mapping in priorities.get("categorical", {}).items():
        combo = tuple(float(cf[i]) for i in group_indices)
        weight = mapping.get(combo, 0)
        scores.append(float(weight) if weight is not None else 0.0)
    return float(np.sum(scores)) if scores else 1.0


def max_attainable_priority(
    priorities: Dict[str, Any], grid_size: int = 1001, refine_steps: int = 40,
) -> Optional[float]:
    """Upper bound of :func:`compute_priority_score` over the allowed region.

    Sums each actionable numerical feature's maximum priority over its
    ``[min, max]`` box and each categorical group's largest allowed weight.
    The per-feature maximum is a grid scan followed by a golden-section
    refinement around the best grid point, because the priority functions peak
    at a sample-relative anchor that a fixed grid can miss (an underestimate
    would make normalised scores exceed 1). No model calls.
    """
    total = 0.0
    for constraint in priorities.get("numerical", {}).values():
        if not isinstance(constraint, dict) or constraint.get("function") is None:
            continue
        lo = float(constraint["min"])
        hi = float(constraint["max"])
        fn = constraint["function"]

        def evaluate(v: float) -> float:
            try:
                w = float(np.asarray(fn(float(v))).squeeze())
            except Exception:
                return float("-inf")
            return w if np.isfinite(w) else float("-inf")

        if hi <= lo:
            best = evaluate(lo)
            total += best if np.isfinite(best) else 0.0
            continue

        grid = np.linspace(lo, hi, int(grid_size))
        values = np.array([evaluate(v) for v in grid])
        k = int(values.argmax())
        best_v, best_w = float(grid[k]), float(values[k])
        # Refine inside the bracket around the best grid point.
        left = float(grid[max(0, k - 1)])
        right = float(grid[min(len(grid) - 1, k + 1)])
        for _ in range(int(refine_steps)):
            if right - left < 1e-12:
                break
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            w1, w2 = evaluate(m1), evaluate(m2)
            if w1 > best_w:
                best_v, best_w = m1, w1
            if w2 > best_w:
                best_v, best_w = m2, w2
            if w1 >= w2:
                right = m2
            else:
                left = m1
        total += best_w if np.isfinite(best_w) else 0.0
    for mapping in priorities.get("categorical", {}).values():
        allowed = [float(w) for w in mapping.values() if w is not None]
        total += max(allowed) if allowed else 0.0
    return float(total) if total > 0.0 else None


class BasePriorityMethod:
    name = "base"
    supports_multiple = True

    def __init__(self, *, pctx, epsilon: float = 0.05, **params: Any) -> None:
        self.pctx = pctx
        self.epsilon = float(epsilon)
        self.params = dict(params)

    def generate_many(
        self, x: np.ndarray, target: float, priorities: Dict[str, Any], n_cfs: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MINLPMethod(BasePriorityMethod):
    """Single-counterfactual MINLP search."""

    name = "minlp"
    supports_multiple = False

    def generate_many(self, x, target, priorities, n_cfs):
        p = self.params
        explainer = MINLSearchExplainer(
            model_pred=self.pctx.model_predict,
            priorities=priorities,
            sample=np.asarray(x, dtype=float).tolist(),
            target=float(target),
            dataset=np.asarray(self.pctx.X_train, dtype=float).copy(),
            target_exemplar_epsilon=float(p.get("target_exemplar_epsilon", 0.10)),
            epsilon=self.epsilon,
            workflow_logger=workflow_logger,
            feature_names=self.pctx.feature_names,
        )
        error: Optional[str] = None
        cf_raw = None
        try:
            cf_raw = explainer.find_counterfactuals(
                shap_approx=bool(p.get("shap_approx", True)),
                num_samples=int(p.get("shap_num_samples", 200)),
                max_iterations=int(p.get("max_iterations", 10)),
                patience=int(p.get("patience", 5)),
                fallback_random_max_iterations=int(
                    p.get("fallback_random_max_iterations", 10000)),
                peak_lock_in=bool(p.get("peak_lock_in", True)),
                peak_lock_in_max_sweeps=int(p.get("peak_lock_in_max_sweeps", 3)),
                random_seed=p.get("seed", None),
                trust_region=bool(p.get("trust_region", True)),
                trust_region_init=float(p.get("trust_region_init", 0.25)),
                trust_region_min=float(p.get("trust_region_min", 0.02)),
                trust_region_max=float(p.get("trust_region_max", 1.0)),
                residual_correction=bool(p.get("residual_correction", True)),
                restoration=bool(p.get("restoration", True)),
            )
        except Exception as exc:
            error = str(exc)
            logger.warning("MINLP find_counterfactuals failed: %s", exc)
        last = getattr(explainer, "last_search_result", None) or {}
        exemplar_source = getattr(explainer, "exemplar_source", None)
        exemplar_pred_distance = last.get(
            "exemplar_pred_distance",
            getattr(explainer, "exemplar_pred_distance", None))
        warm_start = last.get("warm_start") or {}
        peak = last.get("peak_lock_in") or {}
        step = last.get("step_control") or {}
        cfs: List[np.ndarray] = []
        if cf_raw is not None:
            cfs.append(np.asarray(cf_raw, dtype=float).reshape(-1))
        return {
            "cfs": cfs,
            "cf_iterations": None,
            "iterations": int(last.get("iterations_run", 0)),
            "error": error,
            "extra": {
                "reached_target": bool(last.get("reached_target", False)),
                "stop_reason": str(last.get("stop_reason", "unknown")),
                "exemplar_source": exemplar_source,
                "exemplar_pred_distance": exemplar_pred_distance,
                "cf_source": last.get("cf_source"),
                "anchor_priority_score": last.get("anchor_priority_score"),
                "priority_gain_vs_anchor": last.get("priority_gain_vs_anchor"),
                "peak_lock_in_moves": peak.get("accepted_moves"),
                "peak_lock_in_gain": peak.get("priority_gain"),
                "peak_lock_in_model_calls": peak.get("model_calls"),
                "search_exception": last.get("search_exception"),
                "warm_start_total_combos": warm_start.get("total_combos"),
                "warm_start_feasible_combos": warm_start.get("feasible_combos"),
                "warm_start_best_model_gap": warm_start.get("best_warmstart_model_gap"),
                "warm_start_best_linear_gap": warm_start.get("best_warmstart_linear_gap"),
                "accepted_steps": step.get("accepted_steps"),
                "rejected_steps": step.get("rejected_steps"),
                "restoration_hits": step.get("restoration_hits"),
                "trust_fraction_final": step.get("trust_fraction_final"),
            },
        }


class RandomSearchMethod(BasePriorityMethod):
    """Random-search baseline that samples from the priority distributions."""

    name = "random_search"
    supports_multiple = True

    def generate_many(self, x, target, priorities, n_cfs):
        p = self.params
        explainer = RandomSearchExplainer(
            model_pred=self.pctx.model_predict,
            priorities=priorities,
            sample=np.asarray(x, dtype=float).tolist(),
            target=float(target),
        )
        samples, preds, scores, iters = explainer.generate_random_samples(
            expected_counterfactuals=int(n_cfs),
            max_iterations=int(p.get("max_iterations", 10000)),
            epsilon=self.epsilon,
            random_seed=p.get("seed", None),
            use_monte_carlo=bool(p.get("use_monte_carlo", True)),
            max_tries=int(p.get("max_tries", 100)),
        )
        cfs = [np.asarray(s, dtype=float).reshape(-1) for s in samples]
        return {
            "cfs": cfs,
            "cf_iterations": [int(i) for i in iters] if iters else None,
            "iterations": None,
            "error": None,
        }


_REGISTRY = {
    MINLPMethod.name: MINLPMethod,
    RandomSearchMethod.name: RandomSearchMethod,
}


def build_method(name: str, *, pctx, epsilon: float = 0.05, **params: Any) -> BasePriorityMethod:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown priority method '{name}'. Available: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name](pctx=pctx, epsilon=epsilon, **params)


__all__ = [
    "BasePriorityMethod",
    "MINLPMethod",
    "RandomSearchMethod",
    "build_method",
    "compute_priority_score",
    "max_attainable_priority",
]
