"""Run the standard counterfactual baselines inside the paper harness.

``standard_methods`` has its own API (``generate_many(x, target, bounds,
features_to_vary, n_cfs)``) and no notion of priorities. This bridge wraps each
of them in the priority-method interface so a study config can list them next
to ``minlp`` and ``random_search``, and so every method is scored with the same
metrics -- including ``priority_score``, computed post hoc on whatever
counterfactual the baseline returned.

Fairness of the comparison rests on one thing: the baselines get the **same
feasible region** the priority methods get. It is derived here from the
priority set:

* a numerical feature with ``function=None`` is pinned to the sample's value,
* an actionable numerical feature may move inside ``[min, max]``,
* categorical one-hot columns are pinned to the sample's category.

The last point is a deliberate PoC simplification: the baselines have no
one-hot repair step, so letting them move those columns would produce invalid
states. It makes the comparison conservative for the priority methods, which
*can* switch categories -- report it alongside the results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from explainit.experiments.continuous_minlp.priority_methods.methods import (
    BasePriorityMethod,
    build_method as build_priority_method,
)
from explainit.experiments.continuous_minlp.priority_methods.selection import CountedModel
from explainit.experiments.continuous_minlp.standard_methods import methods as std

logger = logging.getLogger("explainit.experiments.paper.standard_bridge")


def region_from_priorities(
    priorities: Dict[str, Any], x: np.ndarray, *, allow_categorical: bool = False,
) -> Tuple[List[Optional[Tuple[float, float]]], List[int], List[Tuple[Tuple[int, ...], List[int]]]]:
    """``(bounds, features_to_vary, categorical_groups)`` implied by a priority set.

    With ``allow_categorical`` the one-hot columns of a group that permits more
    than one category are released into ``[0, 1]`` and returned as a group, so
    the caller can repair the method's output into a valid one-hot state;
    forbidden categories stay pinned.
    """
    x = np.asarray(x, dtype=float)
    bounds: List[Optional[Tuple[float, float]]] = [
        (float(v), float(v)) for v in x
    ]
    vary: List[int] = []
    for idx, cfg in priorities.get("numerical", {}).items():
        if not isinstance(cfg, dict) or cfg.get("function") is None:
            continue
        bounds[int(idx)] = (float(cfg["min"]), float(cfg["max"]))
        vary.append(int(idx))

    groups: List[Tuple[Tuple[int, ...], List[int]]] = []
    if allow_categorical:
        for group_indices, mapping in priorities.get("categorical", {}).items():
            allowed = [combo for combo, w in mapping.items()
                       if w is not None and float(w) > 0.0]
            if len(allowed) <= 1:
                continue
            cols = [int(i) for i in group_indices]
            usable = [j for j in range(len(cols))
                      if any(float(combo[j]) > 0.5 for combo in allowed)]
            if len(usable) <= 1:
                continue
            for j in usable:
                bounds[cols[j]] = (0.0, 1.0)
                vary.append(cols[j])
            groups.append((tuple(cols), usable))
    return bounds, sorted(set(vary)), groups


def repair_one_hot(cf: np.ndarray,
                   groups: Sequence[Tuple[Tuple[int, ...], List[int]]]) -> np.ndarray:
    """Snap each released categorical group back to a single active category."""
    out = np.asarray(cf, dtype=float).copy()
    for cols, usable in groups:
        winner = usable[int(np.argmax([out[cols[j]] for j in usable]))]
        for j, col in enumerate(cols):
            out[col] = 1.0 if j == winner else 0.0
    return out


class StandardMethodAdapter(BasePriorityMethod):
    """Expose a ``standard_methods`` algorithm as a priority-branch method."""

    supports_multiple = True

    def __init__(self, *, pctx, epsilon: float = 0.05, std_name: str, **params: Any) -> None:
        super().__init__(pctx=pctx, epsilon=epsilon, **params)
        self.name = std_name
        self._std_name = std_name
        self._inner = None
        self.allow_categorical = bool(self.params.pop("allow_categorical", False))

    def _method(self):
        if self._inner is None:
            model = self.pctx.model
            if model is not None and hasattr(self.pctx.model_predict, "n_rows"):
                # Charge gradient methods for calls they make straight to the
                # network instead of through model_predict.
                model = CountedModel(model, self.pctx.model_predict)
            self._inner = std.build_method(
                self._std_name,
                model=model,
                model_predict=self.pctx.model_predict,
                X_train=np.asarray(self.pctx.X_train, dtype=float),
                feature_names=list(self.pctx.feature_names),
                epsilon=self.epsilon,
                **{k: v for k, v in self.params.items() if k != "seed"},
            )
            self.supports_multiple = getattr(self._inner, "supports_multiple", True)
        return self._inner

    def generate_many(self, x, target, priorities, n_cfs):
        bounds, vary, groups = region_from_priorities(
            priorities, x, allow_categorical=self.allow_categorical)
        if not vary:
            return {"cfs": [], "iterations": None, "cf_iterations": None,
                    "error": "no actionable feature in this priority set",
                    "extra": {}}
        seed = self.params.get("seed")
        if seed is not None:
            np.random.seed(int(seed))
        out = self._method().generate_many(
            np.asarray(x, dtype=float), float(target), bounds, vary, int(n_cfs))
        cfs = [np.asarray(c, dtype=float).reshape(-1)
               for c in (out.get("cfs") or []) if c is not None]
        if groups:
            cfs = [repair_one_hot(c, groups) for c in cfs]
        return {
            "cfs": cfs,
            "iterations": out.get("iterations"),
            "cf_iterations": None,
            "error": out.get("error"),
            "extra": {"n_features_varied": len(vary),
                      "n_categorical_groups_released": len(groups)},
        }


def available_methods() -> List[str]:
    return sorted(set(std.available_methods()) | {"minlp", "random_search"})


def build_any_method(name: str, *, pctx, epsilon: float = 0.05, **params: Any):
    """Build a priority-branch method, falling back to the standard registry."""
    try:
        return build_priority_method(name, pctx=pctx, epsilon=epsilon, **params)
    except KeyError:
        pass
    if name not in std.METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {available_methods()}")
    return StandardMethodAdapter(
        pctx=pctx, epsilon=epsilon, std_name=name, **params)


__all__ = [
    "StandardMethodAdapter",
    "available_methods",
    "build_any_method",
    "region_from_priorities",
    "repair_one_hot",
]
