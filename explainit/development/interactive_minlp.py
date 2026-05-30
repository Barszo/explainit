"""Per-sample MINLP workbench.

This module is intended for hand-driven exploration of
``MINLSearchExplainer``: pick a dataset, pick one or more concrete samples,
write your own priorities, then run MINLP search and inspect the output for
each sample individually.

Two ways to use it:

1. **Edit-and-run script** (recommended for quick iteration).
   Modify the ``USER_*`` constants and ``build_my_priorities`` function near
   the bottom of this file and run::

       python -m explainit.development.interactive_minlp

   The script will dispatch each chosen sample to ``inspect_sample`` and
   print a rich per-sample trace using the helpers from ``inspect_minlp``.

2. **Library use** (notebook / REPL)::

       from explainit.development.interactive_minlp import (
           load_context, show_features, show_sample, find_target_exemplar,
           PriorityBuilder, run_minlp_on_sample,
       )

       ctx = load_context("german_credit")
       show_features(ctx)
       show_sample(ctx, 0)

       target = find_target_exemplar(ctx, target_class=1, target_probability=0.75)

       pb = PriorityBuilder(ctx)
       pb.add_exponential("duration_in_month", x0=12, x1=6, increasing=False)
       pb.add_linear("credit_amount",  x0=10_000, x1=3_000, increasing=False)
       pb.set_non_actionable("age_in_years")
       priorities = pb.build()

       run_minlp_on_sample(ctx, sample_index=0, priorities=priorities,
                           target_class=1, target_probability=0.75)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEV_DIR = Path(__file__).resolve().parent
if str(DEV_DIR.parents[1]) not in sys.path:
    sys.path.insert(0, str(DEV_DIR.parents[1]))

from explainit.priorities.linear import basic_linear as _basic_linear  # noqa: E402
from explainit.priorities.nonlinear import exponential as _exponential  # noqa: E402

from explainit.development.inspect_minlp import (  # noqa: E402
    DATASET_FILES,
    DevContext,
    describe_cf,
    describe_priorities,
    describe_shapley,
    evaluate_cf,
    inspect_sample as _inspect_sample,
    load_context,
    model_predict_fn,
    select_sample_indices,
)
from explainit.explainers.minlp_search import MINLSearchExplainer  # noqa: E402


logger = logging.getLogger("explainit.development.interactive_minlp")


# ---------------------------------------------------------------------------
# Context-aware inspection helpers
# ---------------------------------------------------------------------------


def show_features(ctx: DevContext) -> None:
    """Print the index -> feature name table for the loaded dataset."""

    logger.info("Features in dataset '%s' (%d total):", ctx.dataset_key, len(ctx.feature_names))
    for idx, name in enumerate(ctx.feature_names):
        logger.info("  %3d  %s", idx, name)


def _as_array(arr: Any) -> np.ndarray:
    return arr.values if hasattr(arr, "values") else np.asarray(arr)


def show_sample(ctx: DevContext, sample_index: int, max_features: int = 25) -> np.ndarray:
    """Print the feature values for a test-set sample and return them."""

    X_test = _as_array(ctx.X_test)
    if sample_index < 0 or sample_index >= len(X_test):
        raise IndexError(
            f"sample_index {sample_index} out of range (test size {len(X_test)})"
        )
    y_test = _as_array(ctx.y_test).flatten()
    sample = X_test[sample_index].astype(float)
    pred = float(ctx.model.predict(sample.reshape(1, -1), verbose=0)[0, 0])
    logger.info(
        "Sample %d | true label=%d | model p(class=1)=%.4f",
        sample_index, int(y_test[sample_index]), pred,
    )
    for idx in range(min(len(sample), max_features)):
        logger.info("  %3d  %-24s = %.4f", idx, ctx.feature_names[idx], sample[idx])
    if len(sample) > max_features:
        logger.info("  ... %d more features not shown ...", len(sample) - max_features)
    return sample


def find_target_exemplar(
    ctx: DevContext,
    *,
    target_class: int,
    target_probability: float = 0.75,
) -> np.ndarray:
    """Pick a training-set point whose model probability is close to target."""

    X = _as_array(ctx.X_train)
    y = _as_array(ctx.y_train).flatten()
    mask = y == target_class
    if not np.any(mask):
        raise ValueError(f"No training samples with class {target_class}.")
    preds = ctx.model.predict(X[mask], verbose=0).flatten()
    idx_local = int(np.argmin(np.abs(preds - target_probability)))
    target = X[mask][idx_local].astype(float)
    logger.info(
        "Picked target exemplar with model p(class=1)=%.4f (requested %.2f).",
        float(preds[idx_local]), target_probability,
    )
    return target


# ---------------------------------------------------------------------------
# Priority builder (numerical + categorical)
# ---------------------------------------------------------------------------


def _resolve_feature(ctx: DevContext, feature: Any) -> int:
    """Accept either an integer index or a feature-name string."""

    if isinstance(feature, (int, np.integer)):
        idx = int(feature)
    else:
        try:
            idx = ctx.feature_names.index(str(feature))
        except ValueError as exc:
            raise KeyError(
                f"Feature '{feature}' not found in dataset '{ctx.dataset_key}'."
            ) from exc
    if idx < 0 or idx >= len(ctx.feature_names):
        raise IndexError(f"Feature index {idx} out of range.")
    return idx


class PriorityBuilder:
    """Fluent builder for the priorities dict expected by MINLSearchExplainer.

    All ``add_*`` / ``set_*`` methods return ``self`` so the calls chain
    naturally. ``build()`` returns the priorities dict and (optionally) a
    list of human-readable specifications for logging.

    Bounds defaults:
      * numerical features get ``[min, max]`` from ``X_train`` unless you
        pass ``min_val`` / ``max_val`` overrides;
      * any feature you don't touch is treated as actionable with the
        training-data range and a constant 0.5 weight, so MINLP will still
        let it move but won't prefer one direction over another.
    """

    def __init__(self, ctx: DevContext, sample: Optional[np.ndarray] = None):
        self.ctx = ctx
        self._X_train_np = _as_array(ctx.X_train)
        self._n_features = self._X_train_np.shape[1]
        if sample is None:
            self.sample = None
        else:
            self.sample = np.asarray(sample, dtype=float).flatten()
        self._numerical: Dict[int, Dict[str, Any]] = {}
        self._categorical: Dict[Tuple[int, ...], Dict[Tuple[float, ...], Optional[float]]] = {}
        self._descriptions: List[str] = []

    # ---- numerical helpers -------------------------------------------------

    def add_numerical(
        self,
        feature: Any,
        function: Callable[[float], float],
        *,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "PriorityBuilder":
        idx = _resolve_feature(self.ctx, feature)
        dmin = float(self._X_train_np[:, idx].min())
        dmax = float(self._X_train_np[:, idx].max())
        lo = dmin if min_val is None else float(min_val)
        hi = dmax if max_val is None else float(max_val)
        if lo > hi:
            lo, hi = hi, lo
        self._numerical[idx] = {"min": lo, "max": hi, "function": function}
        self._descriptions.append(
            f"NUM custom  idx={idx:>3} {self.ctx.feature_names[idx]:<24s} bounds=[{lo:.4f},{hi:.4f}]"
        )
        return self

    def add_linear(
        self,
        feature: Any,
        *,
        x0: float,
        x1: float,
        increasing: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "PriorityBuilder":
        def _fn(x, _x0=float(x0), _x1=float(x1), _inc=bool(increasing)):
            return float(np.asarray(_basic_linear(x, x0=_x0, x1=_x1, increasing=_inc)).squeeze())
        return self.add_numerical(feature, _fn, min_val=min_val, max_val=max_val)

    def add_exponential(
        self,
        feature: Any,
        *,
        x0: float,
        x1: float,
        increasing: bool = True,
        a: float = 5.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "PriorityBuilder":
        def _fn(x, _x0=float(x0), _x1=float(x1), _inc=bool(increasing), _a=float(a)):
            return float(np.asarray(_exponential(x, x0=_x0, x1=_x1, increasing=_inc, a=_a)).squeeze())
        return self.add_numerical(feature, _fn, min_val=min_val, max_val=max_val)

    def add_constant_pref(
        self,
        feature: Any,
        weight: float = 0.5,
        *,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "PriorityBuilder":
        w = float(weight)

        def _fn(_x, _w=w):
            return _w

        return self.add_numerical(feature, _fn, min_val=min_val, max_val=max_val)

    def set_non_actionable(self, feature: Any) -> "PriorityBuilder":
        idx = _resolve_feature(self.ctx, feature)
        if self.sample is None:
            raise RuntimeError(
                "Cannot mark a feature non-actionable without a sample. "
                "Pass `sample=...` when constructing PriorityBuilder."
            )
        value = float(self.sample[idx])
        self._numerical[idx] = {"min": value, "max": value, "function": None}
        self._descriptions.append(
            f"NUM fixed   idx={idx:>3} {self.ctx.feature_names[idx]:<24s} value={value:.4f}"
        )
        return self

    # ---- categorical helpers ----------------------------------------------

    def add_categorical(
        self,
        features: Sequence[Any],
        weights: Dict[Tuple[float, ...], Optional[float]],
    ) -> "PriorityBuilder":
        group = tuple(_resolve_feature(self.ctx, f) for f in features)
        clean: Dict[Tuple[float, ...], Optional[float]] = {}
        for combo, weight in weights.items():
            combo_tuple = tuple(float(v) for v in combo)
            clean[combo_tuple] = None if weight is None else float(weight)
        self._categorical[group] = clean
        self._descriptions.append(
            f"CAT group idx={list(group)} | {len(clean)} combinations"
        )
        return self

    # ---- finalisation ------------------------------------------------------

    def _fill_missing_numerical(self) -> None:
        """For untouched features, install a permissive neutral preference."""
        for idx in range(self._n_features):
            if idx in self._numerical:
                continue
            dmin = float(self._X_train_np[:, idx].min())
            dmax = float(self._X_train_np[:, idx].max())

            def _flat(_x, _w=0.5):
                return _w

            self._numerical[idx] = {"min": dmin, "max": dmax, "function": _flat}
        # No automatic categorical defaults; absence == no categorical group.

    def build(self) -> Dict[str, Any]:
        self._fill_missing_numerical()
        return {"numerical": dict(self._numerical), "categorical": dict(self._categorical)}

    def describe(self) -> List[str]:
        return list(self._descriptions)


# ---------------------------------------------------------------------------
# Per-sample MINLP runner
# ---------------------------------------------------------------------------


def run_minlp_on_sample(
    ctx: DevContext,
    sample_index: int,
    priorities: Dict[str, Any],
    *,
    num_cfs: int = 2,
    target_class: int = 1,
    target_probability: float = 0.75,
    threshold: float = 0.5,
    epsilon: float = 0.1,
    target_exemplar_epsilon: float = 0.05,
    max_iterations: int = 200,
    shap_approx: bool = True,
    shap_num_samples: int = 200,
) -> Dict[str, Any]:
    """Run MINLSearchExplainer on one sample with user-supplied priorities.

    Returns a dict with the original sample, the CFs returned, model
    predictions, validity flags, and timing.
    """

    X_test = _as_array(ctx.X_test)
    sample = X_test[sample_index].astype(float)
    original_pred = float(ctx.model.predict(sample.reshape(1, -1), verbose=0)[0, 0])
    original_class = int(original_pred >= threshold)

    logger.info(
        "Running MINLP on dataset=%s sample=%d (orig_pred=%.4f, orig_class=%d -> target_class=%d)",
        ctx.dataset_key, sample_index, original_pred, original_class, target_class,
    )

    describe_priorities(priorities, ctx.feature_names, sample)

    X_train_np = _as_array(ctx.X_train)
    explainer = MINLSearchExplainer(
        model_pred=model_predict_fn(ctx.model),
        priorities=priorities,
        sample=sample.tolist(),
        target=float(target_probability),
        dataset=X_train_np.copy(),
        target_exemplar_epsilon=float(target_exemplar_epsilon),
        epsilon=float(epsilon),
    )

    import time
    started = time.perf_counter()
    cfs: List[np.ndarray] = []
    predictions: List[float] = []
    metrics_list: List[Dict[str, Any]] = []
    error: Optional[str] = None
    try:
        out = explainer.find_counterfactuals_for_binary(
            target_class=int(target_class),
            threshold=float(threshold),
            expected_counterfactuals=int(num_cfs),
            max_iterations=int(max_iterations),
            shap_approx=bool(shap_approx),
            num_samples=int(shap_num_samples),
            return_top_n=int(num_cfs),
        )
        raw_cfs, raw_preds, raw_scores, raw_iters = out
        for rank, (cf, pred, score, iters) in enumerate(
            zip(raw_cfs, raw_preds, raw_scores, raw_iters), start=1
        ):
            cf_arr = np.asarray(cf, dtype=float).flatten()
            metrics = evaluate_cf(
                ctx, sample, cf_arr, threshold=threshold, target_class=target_class,
            )
            metrics["explainer_prediction"] = float(pred)
            metrics["preference_score_internal"] = float(score)
            metrics["found_iteration"] = int(iters)
            describe_cf(
                ctx.feature_names, sample, cf_arr, metrics,
                rank=rank, threshold=threshold, target_class=target_class,
            )
            cfs.append(cf_arr)
            predictions.append(float(pred))
            metrics_list.append(metrics)
    except Exception as exc:
        logger.exception("MINLP search raised: %s", exc)
        error = str(exc)
    elapsed = time.perf_counter() - started

    try:
        describe_shapley(explainer, ctx.feature_names)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Could not describe Shapley values: %s", exc)

    n_valid = sum(1 for m in metrics_list if m["target_achieved"])
    logger.info(
        "Sample %d done | valid %d/%d | elapsed=%.2fs%s",
        sample_index, n_valid, len(metrics_list), elapsed,
        f" | error={error}" if error else "",
    )
    return {
        "sample_index": sample_index,
        "sample": sample,
        "original_prediction": original_pred,
        "counterfactuals": cfs,
        "predictions": predictions,
        "metrics": metrics_list,
        "elapsed_seconds": elapsed,
        "error": error,
    }


def run_minlp_on_samples(
    ctx: DevContext,
    sample_indices: Iterable[int],
    priorities_builder: Callable[[DevContext, np.ndarray], Dict[str, Any]],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run MINLP per sample with priorities built fresh for each sample.

    ``priorities_builder(ctx, sample)`` must return the priorities dict.
    This lets you mix sample-dependent priorities (e.g. non-actionable
    features fixed to the current sample value) with shared global rules.
    """

    X_test = _as_array(ctx.X_test)
    results: List[Dict[str, Any]] = []
    for idx in sample_indices:
        sample = X_test[int(idx)].astype(float)
        priorities = priorities_builder(ctx, sample)
        results.append(
            run_minlp_on_sample(ctx, int(idx), priorities, **kwargs)
        )
    return results


# ---------------------------------------------------------------------------
# Example / template -- edit the USER_* constants and build_my_priorities
# ---------------------------------------------------------------------------


USER_DATASET = "german_credit"
USER_SAMPLE_INDICES: Sequence[int] = (0,)
USER_TARGET_CLASS = 1
USER_TARGET_PROBABILITY = 0.75
USER_THRESHOLD = 0.5
USER_NUM_CFS = 2
USER_EPSILON = 0.1
USER_TARGET_EXEMPLAR_EPSILON = 0.05
USER_MAX_ITERATIONS = 200
USER_SHAP_APPROX = True
USER_SHAP_NUM_SAMPLES = 200


def build_my_priorities(ctx: DevContext, sample: np.ndarray) -> Dict[str, Any]:
    """Define your priorities here. Called once per sample.

    The default implementation is a small *example* that nudges a few common
    features toward the target exemplar. Replace it freely.
    """

    target = find_target_exemplar(
        ctx,
        target_class=USER_TARGET_CLASS,
        target_probability=USER_TARGET_PROBABILITY,
    )
    pb = PriorityBuilder(ctx, sample=sample)

    for idx, name in enumerate(ctx.feature_names):
        try:
            x0 = float(sample[idx])
            x1 = float(target[idx])
            if abs(x1 - x0) < 1e-9:
                pb.add_constant_pref(idx, weight=0.5)
            else:
                pb.add_exponential(
                    idx, x0=x0, x1=x1, increasing=(x1 > x0), a=5.0,
                )
        except Exception as exc:
            logger.warning("Could not set preference for %s: %s", name, exc)

    return pb.build()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-sample MINLP workbench (uses USER_* defaults from this file)."
    )
    parser.add_argument(
        "--dataset", "-d", choices=sorted(DATASET_FILES), default=USER_DATASET,
        help="Dataset to load from development/data.",
    )
    parser.add_argument(
        "--sample-index", "-i", type=int, action="append", default=None,
        help="Sample index to inspect (repeatable). Overrides USER_SAMPLE_INDICES.",
    )
    parser.add_argument(
        "--target-class", type=int, default=USER_TARGET_CLASS,
    )
    parser.add_argument(
        "--target-probability", type=float, default=USER_TARGET_PROBABILITY,
    )
    parser.add_argument("--threshold", type=float, default=USER_THRESHOLD)
    parser.add_argument("--num-cfs", "-k", type=int, default=USER_NUM_CFS)
    parser.add_argument("--epsilon", type=float, default=USER_EPSILON)
    parser.add_argument(
        "--target-exemplar-epsilon", type=float,
        default=USER_TARGET_EXEMPLAR_EPSILON,
    )
    parser.add_argument(
        "--max-iterations", type=int, default=USER_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--shap-approx", action="store_true", default=USER_SHAP_APPROX,
    )
    parser.add_argument(
        "--shap-exact", dest="shap_approx", action="store_false",
    )
    parser.add_argument(
        "--shap-num-samples", type=int, default=USER_SHAP_NUM_SAMPLES,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--show-features", action="store_true",
        help="Print the feature index/name table and exit.",
    )
    parser.add_argument(
        "--show-sample", type=int, default=None,
        help="Print the values of a specific test sample and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    ctx = load_context(
        args.dataset,
        target_probability=args.target_probability,
        target_class=args.target_class,
    )

    if args.show_features:
        show_features(ctx)
        return
    if args.show_sample is not None:
        show_sample(ctx, args.show_sample)
        return

    sample_indices = (
        list(args.sample_index)
        if args.sample_index
        else list(USER_SAMPLE_INDICES)
    )
    if not sample_indices:
        sample_indices = select_sample_indices(ctx, random_samples=1)

    results = run_minlp_on_samples(
        ctx,
        sample_indices,
        build_my_priorities,
        num_cfs=args.num_cfs,
        target_class=args.target_class,
        target_probability=args.target_probability,
        threshold=args.threshold,
        epsilon=args.epsilon,
        target_exemplar_epsilon=args.target_exemplar_epsilon,
        max_iterations=args.max_iterations,
        shap_approx=args.shap_approx,
        shap_num_samples=args.shap_num_samples,
    )

    total_cfs = sum(len(r["metrics"]) for r in results)
    total_valid = sum(
        sum(1 for m in r["metrics"] if m["target_achieved"]) for r in results
    )
    rate = (100.0 * total_valid / total_cfs) if total_cfs else 0.0
    logger.info(
        "Workbench done: %d sample(s), %d CF(s), %d valid (%.1f%%).",
        len(results), total_cfs, total_valid, rate,
    )


if __name__ == "__main__":
    main()
