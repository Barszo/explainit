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
# Sample / dataset-relative priorities
# ---------------------------------------------------------------------------
#
# The helpers above return *static* functions ``f(value) -> [0, 1]`` that know
# nothing about the sample being explained or the dataset. Many realistic
# preferences are relative -- e.g. "prefer values just below the sample's
# current value" or "decay toward the dataset maximum". Those are expressed
# with an :class:`Anchor` (a point resolved per sample/feature at build time)
# and a :class:`ContextualPriority` (a function factory that receives a
# :class:`FeatureContext`).
#
# Units: features are stored scaled. An :class:`Anchor` ``offset`` is an
# absolute shift in that scaled space, ``pct`` is a fraction of the feature's
# dataset range (``dataset_max - dataset_min``), and ``sample_pct`` is a
# fraction of the sample's absolute feature value. Because scaling is linear,
# ``pct`` matches the same fraction of the raw feature range too.


@dataclass
class FeatureContext:
    """Per-feature, per-sample context passed to a :class:`ContextualPriority`."""

    feature_name: str
    sample_value: float
    dataset_min: float
    dataset_max: float

    @property
    def feature_range(self) -> float:
        return float(self.dataset_max - self.dataset_min)


@dataclass
class Anchor:
    """A point on a feature axis, resolved against a :class:`FeatureContext`.

    ``base`` selects the reference (the sample value, dataset min/max, or a
    literal ``value``); ``offset`` shifts it by an absolute amount, ``pct`` by
    a fraction of the feature's dataset range, and ``sample_pct`` by a
    fraction of the sample's absolute feature value.
    """

    base: str
    offset: float = 0.0
    pct: float = 0.0
    sample_pct: float = 0.0
    value: float = 0.0

    def resolve(self, fc: "FeatureContext") -> float:
        if self.base == "sample":
            base_v = fc.sample_value
        elif self.base == "min":
            base_v = fc.dataset_min
        elif self.base == "max":
            base_v = fc.dataset_max
        elif self.base == "value":
            base_v = self.value
        else:
            raise ValueError(f"Unknown anchor base {self.base!r}.")
        sample_shift = self.sample_pct * abs(fc.sample_value)
        return float(base_v + self.offset + self.pct * fc.feature_range + sample_shift)


def at_sample(
    offset: float = 0.0,
    pct: float = 0.0,
    sample_pct: float = 0.0,
) -> Anchor:
    """Anchor at the sample's value.

    ``offset`` is in scaled units, ``pct`` is a fraction of the dataset range,
    and ``sample_pct`` is a fraction of the sample's absolute value.
    """
    return Anchor(
        "sample",
        offset=float(offset),
        pct=float(pct),
        sample_pct=float(sample_pct),
    )


def at_min(offset: float = 0.0, pct: float = 0.0) -> Anchor:
    """Anchor at the dataset minimum."""
    return Anchor("min", offset=float(offset), pct=float(pct))


def at_max(offset: float = 0.0, pct: float = 0.0) -> Anchor:
    """Anchor at the dataset maximum."""
    return Anchor("max", offset=float(offset), pct=float(pct))


def at_value(value: float) -> Anchor:
    """Anchor at a literal (scaled) value."""
    return Anchor("value", value=float(value))


def _as_anchor(a: Any) -> Anchor:
    if isinstance(a, Anchor):
        return a
    return at_value(float(a))


def _exponential_out(
    x: Any,
    *,
    x0: float,
    x1: float,
    increasing: bool = True,
    a: float = 5.0,
) -> np.ndarray:
    """Exponential ease-out transition.

    Compared with :func:`_exponential`, this variant changes quickly near
    ``x0`` and then flattens as it approaches the target value near ``x1``.
    """

    arr = np.asarray(x, dtype=float)
    if x1 <= x0:
        return np.zeros_like(arr) if increasing else np.ones_like(arr)
    t = np.clip((arr - x0) / (x1 - x0), 0.0, 1.0)
    curve = (1.0 - np.exp(-float(a) * t)) / (1.0 - np.exp(-float(a)))
    if increasing:
        return np.where(arr <= x0, 0.0, np.where(arr >= x1, 1.0, curve))
    return np.where(arr <= x0, 1.0, np.where(arr >= x1, 0.0, 1.0 - curve))


def _shaped_transition(
    x: Any,
    *,
    x0: float,
    x1: float,
    shape: str,
    increasing: bool,
    a: float,
) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if shape == "exponential":
        return _exponential(arr, x0=x0, x1=x1, increasing=increasing, a=a)
    if shape == "exponential_out":
        return _exponential_out(arr, x0=x0, x1=x1, increasing=increasing, a=a)
    return _basic_linear(arr, x0=x0, x1=x1, increasing=increasing)


def _shift_boundary_anchor(
    anchor_value: Optional[float],
    *,
    raw_peak: float,
    clamped_peak: float,
    boundary: float,
) -> Optional[float]:
    if anchor_value is None:
        return None
    overflow = raw_peak - clamped_peak
    if abs(overflow) <= 1e-12:
        return float(anchor_value)
    if abs(anchor_value - boundary) > 1e-12:
        return float(anchor_value)
    return float(anchor_value + overflow)


@dataclass
class ContextualPriority:
    """A priority function that is materialised per sample.

    ``build(FeatureContext) -> f(value) -> [0, 1]``. Optional bound overrides
    behave like :func:`numerical_entry`.
    """

    build: Callable[["FeatureContext"], PriorityFn]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    use_dataset_bounds: bool = True
    plot_points: Optional[Callable[["FeatureContext"], Sequence[float]]] = None


def peak_priority(
    *,
    peak_at: Any,
    peak_value: float = 1.0,
    left: Optional[Any] = None,
    left_shape: str = "linear",
    right: Optional[Any] = None,
    right_shape: str = "linear",
    a: float = 5.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    use_dataset_bounds: bool = True,
) -> ContextualPriority:
    """Preference that peaks at ``peak_at`` and decays to 0 on each side.

    * ``peak_at`` / ``left`` / ``right`` are :class:`Anchor` objects (or plain
      numbers). ``peak_value`` is the height at the peak.
    * ``left`` is where the rising side reaches 0 (``left < peak_at``); ``None``
      means a hard cutoff -- the priority is 0 for values below the peak.
    * ``right`` is where the falling side reaches 0 (``right > peak_at``);
      ``None`` means a hard cutoff -- the priority is 0 above the peak.
    * ``left_shape`` / ``right_shape`` are ``"linear"``, ``"exponential"``,
      or ``"exponential_out"`` (``a`` controls exponential steepness).

    Examples::

        # 0 below the sample, decaying exponentially to 0 at the dataset max.
        peak_priority(peak_at=at_sample(offset=0.5), peak_value=1.0,
                      right=at_max(), right_shape="exponential")

        # peak 0.5 at the sample, linearly down to 0 at 20% below the sample's
        # absolute value, and 0 above the sample.
        peak_priority(peak_at=at_sample(), peak_value=0.5,
                      left=at_sample(sample_pct=-0.20), left_shape="linear")
    """

    peak_anchor = _as_anchor(peak_at)
    left_anchor = None if left is None else _as_anchor(left)
    right_anchor = None if right is None else _as_anchor(right)
    pv = float(peak_value)
    a_val = float(a)
    ls = str(left_shape)
    rs = str(right_shape)

    def _build(fc: "FeatureContext") -> PriorityFn:
        raw_px = peak_anchor.resolve(fc)
        px = float(np.clip(raw_px, fc.dataset_min, fc.dataset_max))
        raw_lx = None if left_anchor is None else left_anchor.resolve(fc)
        raw_rx = None if right_anchor is None else right_anchor.resolve(fc)
        lx = raw_lx
        rx = raw_rx
        if raw_px < fc.dataset_min:
            lx = _shift_boundary_anchor(
                raw_lx, raw_peak=raw_px, clamped_peak=px, boundary=fc.dataset_min,
            )
        elif raw_px > fc.dataset_max:
            rx = _shift_boundary_anchor(
                raw_rx, raw_peak=raw_px, clamped_peak=px, boundary=fc.dataset_max,
            )

        def _fn(x, _px=px, _lx=lx, _rx=rx, _pv=pv, _ls=ls, _rs=rs, _a=a_val):
            x = np.asarray(x, dtype=float)
            if _lx is None or _lx >= _px:
                left_vals = np.where(x >= _px, _pv, 0.0)
            else:
                left_vals = _pv * _shaped_transition(
                    x, x0=_lx, x1=_px, shape=_ls, increasing=True, a=_a,
                )

            if _rx is None or _rx <= _px:
                right_vals = np.zeros_like(x)
            else:
                right_vals = _pv * _shaped_transition(
                    x, x0=_px, x1=_rx, shape=_rs, increasing=False, a=_a,
                )

            return np.where(x <= _px, left_vals, right_vals)

        return _fn

    def _plot_points(fc: "FeatureContext") -> Sequence[float]:
        pts = [float(fc.dataset_min), float(fc.dataset_max), float(fc.sample_value)]
        pts.append(peak_anchor.resolve(fc))
        if left_anchor is not None:
            pts.append(left_anchor.resolve(fc))
        if right_anchor is not None:
            pts.append(right_anchor.resolve(fc))
        return pts

    return ContextualPriority(
        build=_build,
        min_val=min_val,
        max_val=max_val,
        use_dataset_bounds=bool(use_dataset_bounds),
        plot_points=_plot_points,
    )


def plateau_priority(
    *,
    low: Any,
    high: Any,
    weight: float = 1.0,
    left: Optional[Any] = None,
    left_shape: str = "linear",
    right: Optional[Any] = None,
    right_shape: str = "linear",
    a: float = 5.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    use_dataset_bounds: bool = True,
) -> ContextualPriority:
    """Constant plateau on ``[low, high]`` with optional decays on each side."""

    low_anchor = _as_anchor(low)
    high_anchor = _as_anchor(high)
    left_anchor = None if left is None else _as_anchor(left)
    right_anchor = None if right is None else _as_anchor(right)
    w = float(weight)
    a_val = float(a)
    ls = str(left_shape)
    rs = str(right_shape)

    def _build(fc: "FeatureContext") -> PriorityFn:
        raw_lo = low_anchor.resolve(fc)
        raw_hi = high_anchor.resolve(fc)
        lo = float(np.clip(raw_lo, fc.dataset_min, fc.dataset_max))
        hi = float(np.clip(raw_hi, fc.dataset_min, fc.dataset_max))
        if lo > hi:
            lo, hi = hi, lo
        raw_lx = None if left_anchor is None else left_anchor.resolve(fc)
        raw_rx = None if right_anchor is None else right_anchor.resolve(fc)
        lx = raw_lx
        rx = raw_rx
        if raw_lo < fc.dataset_min:
            lx = _shift_boundary_anchor(
                raw_lx, raw_peak=raw_lo, clamped_peak=lo, boundary=fc.dataset_min,
            )
        if raw_hi > fc.dataset_max:
            rx = _shift_boundary_anchor(
                raw_rx, raw_peak=raw_hi, clamped_peak=hi, boundary=fc.dataset_max,
            )

        def _fn(x, _lo=lo, _hi=hi, _lx=lx, _rx=rx, _w=w, _ls=ls, _rs=rs, _a=a_val):
            arr = np.asarray(x, dtype=float)
            vals = np.where((arr >= _lo) & (arr <= _hi), _w, 0.0)
            if _lx is not None and _lx < _lo:
                left_mask = (arr >= _lx) & (arr < _lo)
                vals = np.where(
                    left_mask,
                    _w * _shaped_transition(
                        arr, x0=_lx, x1=_lo, shape=_ls, increasing=True, a=_a,
                    ),
                    vals,
                )
            if _rx is not None and _rx > _hi:
                right_mask = (arr > _hi) & (arr <= _rx)
                vals = np.where(
                    right_mask,
                    _w * _shaped_transition(
                        arr, x0=_hi, x1=_rx, shape=_rs, increasing=False, a=_a,
                    ),
                    vals,
                )
            return vals

        return _fn

    def _plot_points(fc: "FeatureContext") -> Sequence[float]:
        pts = [float(fc.dataset_min), float(fc.dataset_max), float(fc.sample_value)]
        pts.extend([low_anchor.resolve(fc), high_anchor.resolve(fc)])
        if left_anchor is not None:
            pts.append(left_anchor.resolve(fc))
        if right_anchor is not None:
            pts.append(right_anchor.resolve(fc))
        return pts

    return ContextualPriority(
        build=_build,
        min_val=min_val,
        max_val=max_val,
        use_dataset_bounds=bool(use_dataset_bounds),
        plot_points=_plot_points,
    )


# ---------------------------------------------------------------------------
# Registry -- EDIT THIS
# ---------------------------------------------------------------------------


# ===========================================================================
# HOW TO EDIT THIS REGISTRY
# ===========================================================================
#
# ``PRIORITY_SETS`` is a plain nested dict:
#
#     PRIORITY_SETS[<dataset_key>][<set_name>] = {
#         "numerical":   {<feature_name>: <priority fn> | NON_ACTIONABLE, ...},
#         "categorical": {<feature_name>: {<code>: <weight>, ...} | NON_ACTIONABLE, ...},
#     }
#
# To *tweak* a priority        -> edit the relevant function/weight below.
# To *add a new set*           -> add a new ``"<set_name>": {...}`` entry under
#                                 the dataset (any name; not limited to set1/set2).
# To *add a new dataset*       -> add a new ``"<dataset_key>": {...}`` top-level
#                                 entry with one or more sets. Every logical
#                                 feature of that dataset must appear exactly
#                                 once (``build_priorities`` enforces this).
#
# Each set is self-contained: sets and datasets are independent, so adding one
# never constrains another. The helpers below are optional conveniences for the
# diabetes example (they keep repeated shapes DRY); a new dataset can define its
# own helpers or just inline the priority functions.
#
# Units: features are standard-scaled. Anchor ``offset`` values (e.g. age's
# +0.5) are absolute shifts in that scaled space; ``pct`` values are fractions
# of the feature's dataset range; ``sample_pct`` values are fractions of the
# sample's absolute feature value (so sample_pct=-0.20 means "20% below the
# current sample value" on the value axis).
#
# PRIORITY RELAXATION INSTRUCTION: Summary of the main relaxation knobs for the
# diabetes example lives here. Shared features (`age`, `bmi`, `bp`, `s6`) are
# edited in `_diabetes_shared_numerical()`, so changing them affects both
# `set1` and `set2`.
# PRIORITY RELAXATION INSTRUCTION: To move a percentage-based point to the
# right, make its `pct` / `peak_pct` less negative (example: `-0.20 -> -0.10`).
# To move it to the left, make it more negative (example: `-0.20 -> -0.30`).
# PRIORITY RELAXATION INSTRUCTION: To move the age threshold right or left,
# edit `offset=0.5` in the `age` definition below. Larger positive values shift
# it right, smaller values shift it left.
# PRIORITY RELAXATION INSTRUCTION: To make an exponential side go down slower,
# lower `a` (example: `5.0 -> 3.0`). To make it go down faster, raise `a`
# (example: `5.0 -> 8.0`).
# PRIORITY RELAXATION INSTRUCTION: `bp` currently uses one shared `a` value for
# both sides in one `peak_priority(...)` call, so editing that `a` relaxes or
# steepens both sides together.
# PRIORITY RELAXATION INSTRUCTION: Serum features (`s1`..`s5`) use the helper
# `_diabetes_serum_priorities(...)`. Their set-specific horizontal shift,
# height, and exponential steepness are controlled in the `set1` / `set2`
# registry entries below via `peak_pct`, `peak_value`, and `a`.


# --- Diabetes helpers ------------------------------------------------------
# The five diabetes blood-serum measurements share the same priority shape;
# only the peak location and height change between sets.
_DIABETES_SERUM_FEATURES = ("s1", "s2", "s3", "s4", "s5")


def _diabetes_shared_numerical() -> Dict[str, Any]:
    """Non-serum numerical priorities reused across diabetes sets.

    Declaring the shared features (``age``, ``bmi``, ``bp``, ``s6``) here once
    keeps them identical across any diabetes set that spreads this helper, so
    the sets only differ where you *want* them to (the serum features). This is
    a convenience, not a requirement -- a set may override or ignore it.

    These follow the original strict requirements: hard cutoffs are preserved
    where requested, and only the serum features differ between ``set1`` and
    ``set2``.
    """
    return {
        # PRIORITY RELAXATION INSTRUCTION: Edit `offset=0.5` here to move the
        # age threshold right or left for both sets. Edit `a=5.0` here to make
        # the right-side exponential drop slower (smaller `a`) or faster
        # (larger `a`).
        # 0 below sample+0.5, then a fast initial drop that flattens toward 0
        # near the dataset max.
        "age": peak_priority(
            peak_at=at_sample(offset=0.5), peak_value=1.0,
            left=None,
            right=at_max(), right_shape="exponential_out", a=5.0,
        ),
        # PRIORITY RELAXATION INSTRUCTION: Edit `pct=-0.20` here to move bmi's
        # left zero-point. Less negative shifts it right, more negative shifts
        # it left. Edit `a=5.0` here to make the left-side exponential drop
        # slower or faster for both sets.
        # 0.5 at the sample, exponentially down to 0 at 20% of the dataset
        # range below the sample, and 0 above the sample.
        "bmi": peak_priority(
            peak_at=at_sample(), peak_value=0.5,
            left=at_sample(pct=-0.20), left_shape="exponential", a=5.0,
            right=None,
        ),
        # PRIORITY RELAXATION INSTRUCTION: Edit `a=5.0` here if you want both
        # sides of `bp` to go down slower (smaller `a`) or faster (larger `a`)
        # in both sets.
        # 0.5 at the sample, dropping quickly away from the sample and then
        # flattening toward 0 on both sides.
        "bp": peak_priority(
            peak_at=at_sample(), peak_value=0.5,
            left=at_min(), left_shape="exponential",
            right=at_max(), right_shape="exponential_out", a=5.0,
        ),
        # Flat, low preference across the whole dataset range.
        "s6": constant_priority(0.1),
    }


def _diabetes_serum_priorities(
    *,
    peak_pct: float,
    peak_value: float,
    a: float,
) -> Dict[str, Any]:
    """Serum priorities for the original strict diabetes setup.

    The peak sits below the sample by a fraction of the dataset range, decays
    exponentially toward the dataset minimum, and is 0 for all values above
    the peak.
    """
    return {
        name: peak_priority(
            peak_at=at_sample(pct=peak_pct),
            peak_value=peak_value,
            left=at_min(),
            left_shape="exponential",
            a=a,
            right=None,
        )
        for name in _DIABETES_SERUM_FEATURES
    }


# --- The registry (edit / extend freely) -----------------------------------
PRIORITY_SETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "diabetes": {
        # set1 and set2 reuse the same non-serum priorities and keep sex
        # non-actionable; they differ only in the serum features s1 (tc),
        # s2 (ldl), s3 (hdl), s4 (tch), s5 (ltg).
        "set1": {
            "numerical": {
                # PRIORITY RELAXATION INSTRUCTION: Edit `peak_pct=-0.20` below
                # to move the serum peaks for `set1`. Less negative shifts them
                # right, more negative shifts them left. Edit `peak_value=0.7`
                # to raise or lower the serum peak height. Edit `a=5.0` to make
                # the left exponential tail slower or faster for all serum
                # features in `set1`.
                **_diabetes_shared_numerical(),
                # Serum peak at sample-20%-of-range (height 0.7), then 0 above.
                **_diabetes_serum_priorities(peak_pct=-0.20, peak_value=0.7, a=5.0),
            },
            "categorical": {
                "sex": NON_ACTIONABLE,   # keep the sample's sex; never change it
            },
        },
        "set2": {
            "numerical": {
                # PRIORITY RELAXATION INSTRUCTION: Edit `peak_pct=-0.40` below
                # to move the serum peaks for `set2`. Less negative shifts them
                # right, more negative shifts them left. Edit `peak_value=0.5`
                # to raise or lower the serum peak height. Edit `a=5.0` to make
                # the left exponential tail slower or faster for all serum
                # features in `set2`.
                **_diabetes_shared_numerical(),
                # Serum peak at sample-40%-of-range (height 0.5), then 0 above.
                **_diabetes_serum_priorities(peak_pct=-0.40, peak_value=0.5, a=5.0),
            },
            "categorical": {
                "sex": NON_ACTIONABLE,
            },
        },
        # To add another diabetes set, copy a block above and rename it, e.g.:
        # "set3": {"numerical": {...}, "categorical": {"sex": NON_ACTIONABLE}},
    },
    # To add another dataset, add a sibling entry keyed by its dataset_key:
    # "my_dataset": {
    #     "default": {
    #         "numerical": {"feat_a": constant_priority(0.5), ...},
    #         "categorical": {"cat_b": {0: 1.0, 1: 0.0}},
    #     },
    # },
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

        if isinstance(entry, ContextualPriority):
            fc = FeatureContext(
                feature_name=fname,
                sample_value=float(sample_arr[idx]),
                dataset_min=dmin,
                dataset_max=dmax,
            )
            fn = entry.build(fc)
            final_min, final_max = _resolve_bounds(
                entry.min_val, entry.max_val, dmin, dmax, entry.use_dataset_bounds,
            )
            payload = {"function": fn, "min": final_min, "max": final_max}
            if entry.plot_points is not None:
                pts = [float(p) for p in entry.plot_points(fc)]
                if pts:
                    payload["plot_min"] = float(min(pts))
                    payload["plot_max"] = float(max(pts))
            numerical[idx] = payload
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

        # Non-actionable categorical: pin to the sample's own category. Only
        # that one-hot combination keeps a (non-None) weight; the rest are set
        # to None, which the search treats as forbidden/non-actionable.
        if weights is NON_ACTIONABLE:
            sample_one_hot = tuple(float(sample_arr[i]) for i in indices)
            mapping = {}
            for code in categories:
                one_hot = tuple(1.0 if c == code else 0.0 for c in categories)
                mapping[one_hot] = 1.0 if one_hot == sample_one_hot else None
            if not any(v is not None for v in mapping.values()):
                raise ValueError(
                    f"Non-actionable categorical '{fname}' did not match the "
                    f"sample's one-hot state {sample_one_hot}; categories={categories}."
                )
            categorical[indices] = mapping
            continue

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
    "FeatureContext",
    "Anchor",
    "ContextualPriority",
    "at_sample",
    "at_min",
    "at_max",
    "at_value",
    "peak_priority",
]
