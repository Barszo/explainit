"""Stage 3: registered priority builders.

A *priority set* is a callable that, given an :class:`ExperimentContext`,
a sample vector and a target prediction value, returns the priorities
dictionary consumed by ``MINLSearchExplainer`` (and ``RandomSearchExplainer``).

This is the file you edit iteratively while exploring a problem: change
the body of the relevant builder, then re-run ``priorities_selection.py``
to see the resulting coverage, plots and closest exemplars.

The registry has the shape::

    PRIORITY_SETS = {
        "<dataset_key>": {
            "<set_name>": builder_fn,
            ...
        },
        ...
    }

Use :func:`get_priority_set` from other modules to retrieve a builder by
``(dataset_key, set_name)``; ``set_name`` defaults to ``"default"``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.priorities.linear import basic_linear as _basic_linear  # noqa: E402
from explainit.priorities.nonlinear import exponential as _exponential  # noqa: E402


logger = logging.getLogger("explainit.experiments.continuos_minlp.priority_sets")


# ---------------------------------------------------------------------------
# Experiment context
# ---------------------------------------------------------------------------


@dataclass
class ExperimentContext:
    """Bundles the inputs a priority builder needs.

    ``model`` is expected to expose a ``predict(X)`` method (Keras-style).
    ``X_train`` is the numpy 2D feature matrix used to derive default
    bounds and to find target exemplars.
    """

    dataset_key: str
    model: Any
    feature_names: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_name: str = "target"


PriorityBuilderFn = Callable[
    [ExperimentContext, np.ndarray, float],
    Dict[str, Any],
]


# ---------------------------------------------------------------------------
# Helpers reused by builders
# ---------------------------------------------------------------------------


def _model_predict(model: Any, X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(model.predict(arr, verbose=0)).flatten()


def find_continuous_exemplar(
    ctx: ExperimentContext, target_y: float,
) -> np.ndarray:
    """Pick the training row whose model prediction is closest to ``target_y``."""
    preds = _model_predict(ctx.model, ctx.X_train)
    idx = int(np.argmin(np.abs(preds - float(target_y))))
    logger.info(
        "Exemplar for target=%.4f: idx=%d, pred=%.4f", target_y, idx, float(preds[idx]),
    )
    return ctx.X_train[idx].astype(float).copy()


def _column_min_max(X: np.ndarray, idx: int) -> tuple:
    return float(np.min(X[:, idx])), float(np.max(X[:, idx]))


def exponential_priority(
    *, x0: float, x1: float, increasing: bool, a: float = 5.0,
) -> Callable[[float], float]:
    def _fn(x, _x0=float(x0), _x1=float(x1), _inc=bool(increasing), _a=float(a)):
        return float(np.asarray(_exponential(x, x0=_x0, x1=_x1, increasing=_inc, a=_a)).squeeze())
    return _fn


def linear_priority(
    *, x0: float, x1: float, increasing: bool,
) -> Callable[[float], float]:
    def _fn(x, _x0=float(x0), _x1=float(x1), _inc=bool(increasing)):
        return float(np.asarray(_basic_linear(x, x0=_x0, x1=_x1, increasing=_inc)).squeeze())
    return _fn


def constant_priority(weight: float = 0.5) -> Callable[[float], float]:
    w = float(weight)
    def _fn(_x):
        return w
    return _fn


def numerical_entry(
    fn: Optional[Callable[[float], float]],
    *,
    min_val: float,
    max_val: float,
) -> Dict[str, Any]:
    return {"function": fn, "min": float(min_val), "max": float(max_val)}


# ---------------------------------------------------------------------------
# Concrete builders
# ---------------------------------------------------------------------------


def build_diabetes_default(
    ctx: ExperimentContext,
    sample: np.ndarray,
    target_y: float,
) -> Dict[str, Any]:
    """Diabetes default: exponential nudge toward the closest exemplar.

    Same shape as the example used in ``development/interactive_minlp_cont.py``
    -- each feature gets an exponential preference whose direction follows
    the exemplar, with a constant 0.5 fallback when the exemplar value
    coincides with the sample value.
    """

    exemplar = find_continuous_exemplar(ctx, target_y)
    X = ctx.X_train
    numerical: Dict[int, Dict[str, Any]] = {}
    for idx in range(len(ctx.feature_names)):
        x0 = float(sample[idx])
        x1 = float(exemplar[idx])
        dmin, dmax = _column_min_max(X, idx)
        if abs(x1 - x0) < 1e-9:
            fn = constant_priority(weight=0.5)
        else:
            fn = exponential_priority(
                x0=x0, x1=x1, increasing=(x1 > x0), a=5.0,
            )
        numerical[idx] = numerical_entry(fn, min_val=dmin, max_val=dmax)
    return {"numerical": numerical, "categorical": {}}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


PRIORITY_SETS: Dict[str, Dict[str, PriorityBuilderFn]] = {
    "diabetes": {
        "default": build_diabetes_default,
    },
}


def get_priority_set(
    dataset_key: str, name: str = "default",
) -> PriorityBuilderFn:
    if dataset_key not in PRIORITY_SETS:
        raise KeyError(
            f"No priority sets registered for dataset '{dataset_key}'. "
            f"Available: {sorted(PRIORITY_SETS)}"
        )
    sets = PRIORITY_SETS[dataset_key]
    if name not in sets:
        raise KeyError(
            f"No priority set '{name}' for dataset '{dataset_key}'. "
            f"Available: {sorted(sets)}"
        )
    return sets[name]


__all__ = [
    "ExperimentContext",
    "PriorityBuilderFn",
    "PRIORITY_SETS",
    "get_priority_set",
    "find_continuous_exemplar",
    "exponential_priority",
    "linear_priority",
    "constant_priority",
    "numerical_entry",
    "build_diabetes_default",
]
