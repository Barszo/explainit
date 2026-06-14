"""Stage 3: declarative priority sets (this is the file you edit).

A *priority set* is a plain, static description of *how desirable* each
feature value is. You do **not** write any search logic here -- you only
declare, per feature:

* numerical features -> a priority *function* ``f(value) -> [0, 1]`` (1.0 =
  most preferred, 0.0 = unacceptable), and
* categorical features -> a *weights* mapping ``{category_code: weight}``
  (relative preference for each category; 0 = forbidden). Categorical
  features are one-hot encoded in ``data_setup.py``; the integer codes here
  index that encoding, and :func:`build_priorities` expands them to one-hot
  states so the search keeps exactly one category active.

The registry has the shape::

    PRIORITY_SETS = {
        "<dataset_key>": {
            "<set_name>": {
                "numerical": {
                    "<feature_name>": <function or NON_ACTIONABLE>,
                    ...
                },
                "categorical": {
                    "<feature_name>": {<category_value>: <weight>, ...},
                    ...
                },
            },
            ...
        },
        ...
    }

Helpers (:func:`linear_priority`, :func:`exponential_priority`,
:func:`constant_priority`, :func:`interval_priority`) build common
priority functions for you; you can also pass any custom ``lambda v: ...``.

Use :func:`build_priorities` (called by the runners and the selection
workbench) to turn a declarative set into the index-keyed dict that the
explainers consume. It resolves the numerical search bounds and validates
that every dataset feature has a priority.

Bounds rule
-----------
Each numerical feature is searched within ``[min, max]``. You may specify
``min``/``max`` per feature via :func:`numerical_entry`; otherwise the
dataset column min/max are used. With ``use_dataset_bounds=True`` (default)
the dataset min/max win on conflict -- i.e. the final bounds are the
*intersection* of your bounds and the dataset range. Set
``use_dataset_bounds=False`` to use your bounds verbatim.

Note: a priority function returning 0 at some value marks that value as
unacceptable even when it lies inside ``[min, max]`` (the search/sampler
never picks zero-weight values).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
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
    """Bundles the inputs needed to materialise a priority set.

    ``model`` is expected to expose a ``predict(X)`` method (Keras-style).
    ``X_train`` is the numpy 2D feature matrix used to derive default
    numerical bounds.

    ``feature_names`` is the *expanded* column list (categorical features are
    one-hot encoded). ``numerical_features`` lists the numerical column names,
    and ``categorical_groups`` maps each original categorical feature name to
    ``{"indices", "categories", "source_values", "columns"}`` describing its
    one-hot block.
    """

    dataset_key: str
    model: Any
    feature_names: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_name: str = "target"
    numerical_features: List[str] = field(default_factory=list)
    categorical_groups: Dict[str, Any] = field(default_factory=dict)


# Sentinel: declare a numerical feature as non-actionable (kept fixed at the
# sample value, never modified by the search).
NON_ACTIONABLE = object()


PriorityFn = Callable[[float], Any]


# ---------------------------------------------------------------------------
# Helpers for building numerical priority functions
# ---------------------------------------------------------------------------


def linear_priority(*, x0: float, x1: float, increasing: bool = True) -> PriorityFn:
    """Linear ramp between ``x0`` and ``x1``.

    ``increasing=True``  -> 0 for ``x <= x0``, ramps up to 1 for ``x >= x1``.
    ``increasing=False`` -> 1 for ``x <= x0``, ramps down to 0 for ``x >= x1``.

    Example: ``linear_priority(x0=10, x1=50, increasing=True)`` is 0 below 10,
    rises linearly to 1 at 50, then stays 1 above 50.
    """

    def _fn(x, _x0=float(x0), _x1=float(x1), _inc=bool(increasing)):
        return _basic_linear(x, x0=_x0, x1=_x1, increasing=_inc)

    return _fn


def exponential_priority(
    *, x0: float, x1: float, increasing: bool = True, a: float = 5.0,
) -> PriorityFn:
    """Exponential ramp between ``x0`` and ``x1`` (``a`` controls steepness)."""

    def _fn(x, _x0=float(x0), _x1=float(x1), _inc=bool(increasing), _a=float(a)):
        return _exponential(x, x0=_x0, x1=_x1, increasing=_inc, a=_a)

    return _fn


def constant_priority(weight: float = 0.5) -> PriorityFn:
    """Flat preference ``weight`` everywhere in the feature's bounds."""

    w = float(weight)

    def _fn(x):
        return np.full_like(np.asarray(x, dtype=float), w)

    return _fn


def interval_priority(*, low: float, high: float, weight: float = 1.0) -> PriorityFn:
    """Box preference: ``weight`` inside ``[low, high]`` and 0 outside.

    Use this to express "only values in this band are acceptable".
    """

    lo = float(low)
    hi = float(high)
    w = float(weight)

    def _fn(x):
        arr = np.asarray(x, dtype=float)
        return np.where((arr >= lo) & (arr <= hi), w, 0.0)

    return _fn


def numerical_entry(
    function: Optional[PriorityFn],
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    use_dataset_bounds: bool = True,
) -> Dict[str, Any]:
    """Wrap a priority function with optional explicit search bounds.

    Use this only when you want to override the default (dataset) bounds for
    a feature; otherwise just put the bare function in the priority set.
    Pass ``function=None`` to mark the feature non-actionable.
    """

    return {
        "function": function,
        "min": min_val,
        "max": max_val,
        "use_dataset_bounds": bool(use_dataset_bounds),
    }


# ---------------------------------------------------------------------------
# Registry -- EDIT THIS
# ---------------------------------------------------------------------------


PRIORITY_SETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "diabetes": {
        # Example set. Diabetes features are standard-scaled, so values are
        # roughly in [-3, 3]. Every feature must appear exactly once, either
        # under "numerical" or "categorical".
        "default": {
            "numerical": {
                "age": constant_priority(0.5),          # no preference, free to move
                "bmi": linear_priority(x0=2.0, x1=-2.0, increasing=False),   # prefer lower bmi
                "bp": linear_priority(x0=2.0, x1=-2.0, increasing=False),    # prefer lower bp
                "s1": constant_priority(0.5),
                "s2": constant_priority(0.5),
                "s3": constant_priority(0.5),
                "s4": constant_priority(0.5),
                "s5": exponential_priority(x0=2.0, x1=-2.0, increasing=False, a=5.0),
                "s6": constant_priority(0.5),
            },
            "categorical": {
                # 'sex' is one-hot encoded with two category codes (0 and 1).
                # Weights are relative preferences (0 = forbidden). Here both
                # are allowed, with code 0 preferred over code 1.
                "sex": {0: 1.0, 1: 0.5},
            },
        },
    },
}


def get_priority_set(dataset_key: str, name: str = "default") -> Dict[str, Any]:
    """Return the declarative priority set for ``(dataset_key, name)``."""

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


# ---------------------------------------------------------------------------
# Conversion to the index-keyed dict consumed by the explainers
# ---------------------------------------------------------------------------


def _resolve_bounds(
    user_min: Optional[float],
    user_max: Optional[float],
    dataset_min: float,
    dataset_max: float,
    use_dataset_bounds: bool,
) -> tuple:
    if use_dataset_bounds:
        lo = dataset_min if user_min is None else max(float(user_min), dataset_min)
        hi = dataset_max if user_max is None else min(float(user_max), dataset_max)
    else:
        lo = dataset_min if user_min is None else float(user_min)
        hi = dataset_max if user_max is None else float(user_max)
    if lo > hi:
        raise ValueError(
            f"Resolved bounds are empty: min={lo} > max={hi}. "
            f"(user=[{user_min}, {user_max}], dataset=[{dataset_min}, {dataset_max}], "
            f"use_dataset_bounds={use_dataset_bounds})"
        )
    return float(lo), float(hi)


def build_priorities(
    ctx: ExperimentContext,
    set_name: str,
    sample: Sequence[float],
) -> Dict[str, Any]:
    """Materialise a declarative priority set into the explainer's dict format.

    Output shape::

        {
            "numerical":   {feature_index: {"function": fn, "min": ..., "max": ...}},
            "categorical": {(col_idx, ...): {(one_hot_tuple): weight}},
        }

    Numerical declarations are keyed by numerical feature name; categorical
    declarations are keyed by the original categorical feature name and map
    each integer category *code* to a weight. Each code is expanded to the
    corresponding one-hot vector over the feature's columns, so the search
    only ever assigns valid one-hot states (the exactly-one-hot constraint).

    Numerical bounds are resolved per :func:`_resolve_bounds`. Non-actionable
    numerical features are fixed at the corresponding ``sample`` value. A
    ``ValueError`` is raised if any feature has no declared priority.
    """

    spec = get_priority_set(ctx.dataset_key, set_name)
    feature_names = list(ctx.feature_names)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    X = np.asarray(ctx.X_train, dtype=float)
    sample_arr = np.asarray(sample, dtype=float).flatten()

    raw_numerical: Dict[str, Any] = dict(spec.get("numerical", {}) or {})
    raw_categorical: Dict[str, Any] = dict(spec.get("categorical", {}) or {})

    numerical_features = list(ctx.numerical_features)
    categorical_groups = dict(ctx.categorical_groups)

    # Backward compatibility: datasets prepared before categorical support
    # carry no metadata -- treat every column as a numerical feature.
    if not numerical_features and not categorical_groups:
        numerical_features = list(feature_names)

    num_feature_set = set(numerical_features)
    cat_feature_set = set(categorical_groups)

    # Reject declarations that are not logical features of this dataset.
    for fname in raw_numerical:
        if fname not in num_feature_set:
            hint = (
                " (it is a categorical feature; declare it under 'categorical')"
                if fname in cat_feature_set else ""
            )
            raise KeyError(
                f"Priority set '{ctx.dataset_key}/{set_name}' lists '{fname}' under "
                f"'numerical', but it is not a numerical feature{hint}. "
                f"Numerical features: {numerical_features}"
            )
    for fname in raw_categorical:
        if fname not in cat_feature_set:
            hint = (
                " (it is a numerical feature; declare it under 'numerical')"
                if fname in num_feature_set else ""
            )
            raise KeyError(
                f"Priority set '{ctx.dataset_key}/{set_name}' lists '{fname}' under "
                f"'categorical', but it is not a categorical feature{hint}. "
                f"Categorical features: {sorted(cat_feature_set)}"
            )

    # Every feature must be declared.
    missing_num = [n for n in numerical_features if n not in raw_numerical]
    missing_cat = [n for n in categorical_groups if n not in raw_categorical]
    if missing_num or missing_cat:
        raise ValueError(
            f"Priority set '{ctx.dataset_key}/{set_name}' is missing priorities. "
            f"Missing numerical: {missing_num}; missing categorical: {missing_cat}. "
            f"Declare each numerical feature with a function or NON_ACTIONABLE, and "
            f"each categorical feature with a {{code: weight}} mapping. To leave a "
            f"numerical feature unchanged by the search, set it to NON_ACTIONABLE."
        )

    numerical: Dict[int, Dict[str, Any]] = {}
    for fname, entry in raw_numerical.items():
        idx = name_to_idx[fname]
        dmin = float(np.min(X[:, idx]))
        dmax = float(np.max(X[:, idx]))

        if entry is NON_ACTIONABLE:
            fixed = float(sample_arr[idx])
            numerical[idx] = {"function": None, "min": fixed, "max": fixed}
            continue

        if callable(entry):
            fn: Optional[PriorityFn] = entry
            user_min: Optional[float] = None
            user_max: Optional[float] = None
            use_ds = True
        elif isinstance(entry, dict):
            fn = entry.get("function")
            user_min = entry.get("min")
            user_max = entry.get("max")
            use_ds = bool(entry.get("use_dataset_bounds", True))
        else:
            raise TypeError(
                f"Numerical priority for '{fname}' must be a function, a "
                f"numerical_entry(...) dict, or NON_ACTIONABLE; got {type(entry)!r}."
            )

        if fn is None:
            fixed = float(user_min if user_min is not None else sample_arr[idx])
            numerical[idx] = {"function": None, "min": fixed, "max": fixed}
            continue

        final_min, final_max = _resolve_bounds(user_min, user_max, dmin, dmax, use_ds)
        numerical[idx] = {"function": fn, "min": final_min, "max": final_max}

    categorical: Dict[tuple, Dict[tuple, float]] = {}
    for fname, weights in raw_categorical.items():
        group = categorical_groups[fname]
        indices = tuple(int(i) for i in group["indices"])
        categories = list(group["categories"])

        if not isinstance(weights, dict) or not weights:
            raise ValueError(
                f"Categorical priority for '{fname}' must be a non-empty mapping "
                f"{{code: weight}}; got {weights!r}."
            )
        unknown = [c for c in weights if c not in categories]
        if unknown:
            raise ValueError(
                f"Categorical priority for '{fname}' references unknown category "
                f"code(s) {unknown}. Valid codes: {categories}."
            )
        missing_codes = [c for c in categories if c not in weights]
        if missing_codes:
            raise ValueError(
                f"Categorical priority for '{fname}' is missing weights for category "
                f"code(s) {missing_codes}. Declare every code (use 0 to forbid one). "
                f"Valid codes: {categories}."
            )

        mapping: Dict[tuple, float] = {}
        for code, weight in weights.items():
            one_hot = tuple(1.0 if c == code else 0.0 for c in categories)
            mapping[one_hot] = float(weight)
        categorical[indices] = mapping

    return {"numerical": numerical, "categorical": categorical}


__all__ = [
    "ExperimentContext",
    "NON_ACTIONABLE",
    "PriorityFn",
    "PRIORITY_SETS",
    "get_priority_set",
    "build_priorities",
    "numerical_entry",
    "linear_priority",
    "exponential_priority",
    "constant_priority",
    "interval_priority",
]
