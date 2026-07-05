"""Sample / target selection and per-feature actionability.

Given the predicted-target dataset and a per-dataset config block, this
module produces:

* the list of ``(sample_id, x, original_prediction, target)`` records, and
* per-sample actionability: a ``bounds`` list (``(lo, hi)`` or ``None`` per
  feature) plus the ``features_to_vary`` index set (immutable features and
  their one-hot columns are pinned).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.continuous_minlp.model_setup import load_model  # noqa: E402
from explainit.experiments.continuous_minlp.standard_methods.predicted_dataset_setup import (  # noqa: E402
    load_predicted_dataset,
)

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.standard_methods.selection"
)

Bounds = List[Optional[Tuple[float, float]]]


@dataclass
class PredictedContext:
    dataset_key: str
    model: Any
    model_predict: Callable[[np.ndarray], np.ndarray]
    feature_names: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    numerical_features: List[str] = field(default_factory=list)
    categorical_groups: Dict[str, Any] = field(default_factory=dict)
    target_name: str = "target"

    @property
    def feat_min(self) -> np.ndarray:
        return self.X_train.min(axis=0)

    @property
    def feat_max(self) -> np.ndarray:
        return self.X_train.max(axis=0)


def _as_matrix(arr) -> np.ndarray:
    return np.asarray(arr.values if hasattr(arr, "values") else arr, dtype=float)


def load_predicted_context(dataset_key: str) -> PredictedContext:
    data = load_predicted_dataset(dataset_key)
    model = load_model(dataset_key)

    def _predict(X: np.ndarray) -> np.ndarray:
        # Call the model directly (much faster than ``.predict`` for the many
        # single-row evaluations the search methods perform).
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.asarray(model(arr, training=False)).reshape(-1)

    return PredictedContext(
        dataset_key=dataset_key,
        model=model,
        model_predict=_predict,
        feature_names=[str(n) for n in data["feature_names"]],
        X_train=_as_matrix(data["X_train"]),
        X_test=_as_matrix(data["X_test"]),
        y_train=np.asarray(data["y_train"], dtype=float).reshape(-1),
        y_test=np.asarray(data["y_test"], dtype=float).reshape(-1),
        numerical_features=[str(n) for n in data.get("numerical_features", [])],
        categorical_groups=dict(data.get("categorical_groups", {}) or {}),
        target_name=str(data.get("target_name", "target")),
    )


# ---------------------------------------------------------------------------
# Sample + target selection
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    sample_id: int
    x: np.ndarray
    original_prediction: float
    target: float


def select_samples(ctx: PredictedContext, cfg: Dict[str, Any]) -> List[SampleRecord]:
    """Select samples and derive their targets per the config block."""

    strategy = str(cfg.get("strategy", "random"))
    n_samples = int(cfg.get("n_samples", 10))
    seed = cfg.get("seed", 42)
    offset = float(cfg.get("target_offset", -0.3))
    floor = float(cfg.get("skip_if_target_below", 0.0))
    ceil = cfg.get("skip_if_target_above", None)

    X_test = ctx.X_test
    n_rows = len(X_test)
    preds = ctx.model_predict(X_test)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        order = [int(i) for i in rng.permutation(n_rows)]
        expected = n_samples
    elif strategy in {"index", "indices"}:
        sample_indices = cfg.get("sample_indices")
        if not sample_indices:
            raise ValueError(
                "Selection strategy 'indices' requires a non-empty 'sample_indices' list."
            )
        order = [int(i) for i in sample_indices]
        expected = len(order)
        # For index-based selection we default to all provided indices.
        if "n_samples" in cfg and cfg.get("n_samples") is not None:
            expected = min(expected, int(cfg["n_samples"]))
    else:
        raise ValueError(
            f"Unsupported selection strategy '{strategy}'. "
            "Use 'random' or 'indices'."
        )

    records: List[SampleRecord] = []
    seen: set[int] = set()
    for idx in order:
        if len(records) >= expected:
            break
        if idx in seen:
            continue
        seen.add(idx)
        if idx < 0 or idx >= n_rows:
            logger.warning(
                "[%s] Ignoring out-of-range sample index %d (valid range: 0..%d).",
                ctx.dataset_key, idx, n_rows - 1,
            )
            continue
        pred = float(preds[idx])
        target = pred + offset
        if target < floor:
            continue
        if ceil is not None and target > float(ceil):
            continue
        records.append(
            SampleRecord(
                sample_id=int(idx),
                x=X_test[idx].astype(float),
                original_prediction=pred,
                target=float(target),
            )
        )

    if len(records) < expected:
        logger.warning(
            "[%s] Only %d/%d samples satisfy selection constraints "
            "(strategy=%s, offset=%.3f, floor=%.3f).",
            ctx.dataset_key, len(records), expected, strategy, offset, floor,
        )
    return records


# ---------------------------------------------------------------------------
# Actionability: bounds + features_to_vary
# ---------------------------------------------------------------------------


def _feature_columns(ctx: PredictedContext, logical_name: str) -> List[int]:
    if logical_name in ctx.numerical_features:
        return [ctx.feature_names.index(logical_name)]
    if logical_name in ctx.categorical_groups:
        return [int(i) for i in ctx.categorical_groups[logical_name]["indices"]]
    if logical_name in ctx.feature_names:
        return [ctx.feature_names.index(logical_name)]
    raise KeyError(
        f"Unknown feature '{logical_name}' for dataset '{ctx.dataset_key}'. "
        f"Numerical: {ctx.numerical_features}; categorical: {list(ctx.categorical_groups)}."
    )


def build_actionability(
    ctx: PredictedContext, cfg: Dict[str, Any], x: np.ndarray,
) -> Tuple[Bounds, List[int]]:
    """Return ``(bounds, features_to_vary)`` for a single sample ``x``.

    Config block shape::

        immutable: [sex]
        bounds:
          age: {direction: increasing}         # lo = x[age], hi = dataset max
          bmi: {direction: decreasing}         # lo = dataset min, hi = x[bmi]
          bp:  {min: -1.0, max: 2.0}           # explicit box
    """

    x = np.asarray(x, dtype=float)
    d = len(ctx.feature_names)
    bounds: Bounds = [None] * d
    vary = set(range(d))

    for name in cfg.get("immutable", []) or []:
        for col in _feature_columns(ctx, name):
            vary.discard(col)

    feat_min, feat_max = ctx.feat_min, ctx.feat_max
    for name, spec in (cfg.get("bounds", {}) or {}).items():
        spec = spec or {}
        for col in _feature_columns(ctx, name):
            if col not in vary:
                continue
            direction = spec.get("direction")
            lo: float
            hi: float
            if direction == "increasing":
                lo, hi = float(x[col]), float(feat_max[col])
            elif direction == "decreasing":
                lo, hi = float(feat_min[col]), float(x[col])
            else:
                lo = float(spec["min"]) if spec.get("min") is not None else float(feat_min[col])
                hi = float(spec["max"]) if spec.get("max") is not None else float(feat_max[col])
            if lo > hi:
                lo, hi = hi, lo
            bounds[col] = (lo, hi)

    return bounds, sorted(vary)


__all__ = [
    "PredictedContext",
    "SampleRecord",
    "load_predicted_context",
    "select_samples",
    "build_actionability",
]
