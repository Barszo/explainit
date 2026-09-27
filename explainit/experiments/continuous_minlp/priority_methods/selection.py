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
from typing import Any, Callable, Dict, List, Optional

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


class BudgetExceeded(RuntimeError):
    """Raised when a method exceeds the model-evaluation budget it was given."""


class ModelCallCounter:
    """Counting proxy around a ``predict`` callable.

    ``n_calls`` counts invocations, ``n_rows`` the total number of feature
    vectors scored. ``n_rows`` is the budget metric to compare methods on,
    since batched methods spend far fewer calls for the same work. When
    ``max_rows`` is set, the proxy raises :class:`BudgetExceeded` as soon as the
    cap is passed, which is what makes a budget-matched comparison possible.
    """

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray],
                 max_rows: Optional[int] = None) -> None:
        self._predict = predict
        self.max_rows = None if max_rows is None else int(max_rows)
        self.n_calls = 0
        self.n_rows = 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        self.n_calls += 1
        self.n_rows += 1 if arr.ndim == 1 else int(arr.shape[0])
        if self.max_rows is not None and self.n_rows > self.max_rows:
            raise BudgetExceeded(
                f"model-evaluation budget exhausted: {self.n_rows} > {self.max_rows} rows")
        return self._predict(arr)

    def reset(self) -> None:
        self.n_calls = 0
        self.n_rows = 0


class CountedModel:
    """Wrap a Keras model so gradient-based methods are charged for their calls.

    Gradient methods reach the network directly (``model(x)`` inside a
    ``GradientTape``) instead of going through ``model_predict``, so without
    this wrapper their budget is reported as ~0. Attribute access is delegated,
    so the wrapped object still behaves like the model for everything else.
    """

    def __init__(self, model, counter: "ModelCallCounter") -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_counter", counter)

    def __call__(self, *args, **kwargs):
        arr = args[0] if args else None
        if arr is not None:
            try:
                shape = tuple(getattr(arr, "shape", ()))
                rows = int(shape[0]) if shape else 1
            except Exception:
                rows = 1
            self._counter.n_calls += 1
            self._counter.n_rows += rows
            if (self._counter.max_rows is not None
                    and self._counter.n_rows > self._counter.max_rows):
                raise BudgetExceeded(
                    f"model-evaluation budget exhausted: "
                    f"{self._counter.n_rows} > {self._counter.max_rows} rows")
        return self._model(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_model"), item)


def counted_context(pctx: PriorityContext, max_rows: Optional[int] = None) -> tuple:
    """Return ``(pctx_copy, counter)`` sharing the context but counting calls."""

    counter = ModelCallCounter(pctx.model_predict, max_rows=max_rows)
    ctx = pctx.ctx
    return PriorityContext(ctx=ctx, model_predict=counter), counter


def select_samples(pctx: PriorityContext, cfg: Dict[str, Any]) -> List[SampleRecord]:
    """Select samples and derive their targets per the config block.

    Strategy is ``indices`` or ``random``. The target is
    ``prediction + offset``, where the offset comes from exactly one of:

    * ``target_offset`` -- absolute, in the dataset's target units;
    * ``target_offset_pred_std`` -- multiples of the model's prediction
      standard deviation on the test set, which makes one config mean the
      same thing on datasets with different target spreads.

    ``skip_if_target_below`` / ``skip_if_target_above`` drop samples whose
    target would leave the observed prediction range.
    """

    strategy = str(cfg.get("strategy", "random"))
    n_samples = int(cfg.get("n_samples", 10))
    seed = cfg.get("seed", 42)
    floor = float(cfg.get("skip_if_target_below", 0.0))
    ceil = cfg.get("skip_if_target_above", None)

    X_test = pctx.X_test
    n_rows = len(X_test)
    preds = pctx.model_predict(X_test)

    std_factor = cfg.get("target_offset_pred_std", None)
    if std_factor is not None:
        offset = float(std_factor) * float(np.std(preds))
        logger.info("[%s] target offset = %.3f x pred std (%.4f) = %.4f",
                    pctx.dataset_key, float(std_factor), float(np.std(preds)), offset)
    else:
        offset = float(cfg.get("target_offset", -0.3))

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
