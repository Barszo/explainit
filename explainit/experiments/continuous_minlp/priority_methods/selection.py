"""Context loading and sample/target selection for the priority branch.

The priority branch runs against the *original* dataset + model (via
``_context.load_context``). Targets are derived exactly like the
standard-methods stage: ``target = model_prediction + target_offset`` in the
MinMax-scaled ``[0, 1]`` space, with optional skip thresholds.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.continuous_minlp._context import load_context  # noqa: E402
from explainit.experiments.continuous_minlp.priority_sets import (  # noqa: E402
    ExperimentContext,
)

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.priority_methods.selection"
)


@dataclass
class PriorityContext:
    """Bundles an :class:`ExperimentContext` with a fast ``predict`` callable."""

    ctx: ExperimentContext
    model_predict: Callable[[np.ndarray], np.ndarray]

    @property
    def dataset_key(self) -> str:
        return self.ctx.dataset_key

    @property
    def feature_names(self) -> List[str]:
        return self.ctx.feature_names

    @property
    def numerical_features(self) -> List[str]:
        return self.ctx.numerical_features

    @property
    def categorical_groups(self) -> Dict[str, Any]:
        return self.ctx.categorical_groups

    @property
    def X_train(self) -> np.ndarray:
        return self.ctx.X_train

    @property
    def X_test(self) -> np.ndarray:
        return self.ctx.X_test

    @property
    def y_train(self) -> np.ndarray:
        return self.ctx.y_train

    @property
    def target_name(self) -> str:
        return self.ctx.target_name

    @property
    def model(self):
        return self.ctx.model


def load_priority_context(dataset_key: str) -> PriorityContext:
    ctx = load_context(dataset_key)
    model = ctx.model

    def _predict(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Direct call is much faster than ``.predict`` for many single rows.
        return np.asarray(model(arr, training=False)).reshape(-1)

    return PriorityContext(ctx=ctx, model_predict=_predict)


@dataclass
class SampleRecord:
    sample_id: int
    x: np.ndarray
    original_prediction: float
    target: float


def select_samples(pctx: PriorityContext, cfg: Dict[str, Any]) -> List[SampleRecord]:
    """Select samples and derive their targets per the config block.

    Mirrors ``standard_methods.selection.select_samples`` (strategy
    ``indices`` or ``random``; ``target = prediction + target_offset`` with
    ``skip_if_target_below`` / ``skip_if_target_above`` thresholds).
    """

    strategy = str(cfg.get("strategy", "random"))
    n_samples = int(cfg.get("n_samples", 10))
    seed = cfg.get("seed", 42)
    offset = float(cfg.get("target_offset", -0.3))
    floor = float(cfg.get("skip_if_target_below", 0.0))
    ceil = cfg.get("skip_if_target_above", None)

    X_test = pctx.X_test
    n_rows = len(X_test)
    preds = pctx.model_predict(X_test)

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
        if "n_samples" in cfg and cfg.get("n_samples") is not None:
            expected = min(expected, int(cfg["n_samples"]))
    else:
        raise ValueError(
            f"Unsupported selection strategy '{strategy}'. Use 'random' or 'indices'."
        )

    records: List[SampleRecord] = []
    seen: set = set()
    for idx in order:
        if len(records) >= expected:
            break
        if idx in seen:
            continue
        seen.add(idx)
        if idx < 0 or idx >= n_rows:
            logger.warning(
                "[%s] Ignoring out-of-range sample index %d (valid range: 0..%d).",
                pctx.dataset_key, idx, n_rows - 1,
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
            pctx.dataset_key, len(records), expected, strategy, offset, floor,
        )
    return records


__all__ = [
    "PriorityContext",
    "SampleRecord",
    "load_priority_context",
    "select_samples",
]
