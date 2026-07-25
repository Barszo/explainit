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
                "search_exception": last.get("search_exception"),
                "warm_start_total_combos": warm_start.get("total_combos"),
                "warm_start_feasible_combos": warm_start.get("feasible_combos"),
                "warm_start_best_model_gap": warm_start.get("best_warmstart_model_gap"),
                "warm_start_best_linear_gap": warm_start.get("best_warmstart_linear_gap"),
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
]
