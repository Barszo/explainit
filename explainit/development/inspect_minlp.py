"""Diagnostic harness for MINLSearchExplainer (self-contained dev project).

This script lives next to its own ``data/`` and ``models/`` directories under
``explainit/development``. It treats the development directory as a stand-alone
project: it does NOT read anything from ``experiments/`` at runtime.

Populate the local caches once with::

    python -m explainit.development.data_setup --source binary_minlp

(or with ``--source train`` to recreate the models from scratch).

Then run the inspector::

    python -m explainit.development.inspect_minlp \\
        --dataset german_credit --sample-index 0 --num-cfs 2 --verbose

    python -m explainit.development.inspect_minlp \\
        --dataset communities_crime --random-samples 5 --shap-exact

You can also import the helpers in a notebook / REPL::

    from explainit.development.inspect_minlp import (
        DevContext, load_context, build_sample_preferences, inspect_sample,
    )
    ctx = load_context("german_credit")
    inspect_sample(ctx, sample_index=0, num_cfs=2)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEV_DIR = Path(__file__).resolve().parent
DEV_DATA_DIR = DEV_DIR / "data"
DEV_MODELS_DIR = DEV_DIR / "models"
DEV_DATA_DIR.mkdir(exist_ok=True)
DEV_MODELS_DIR.mkdir(exist_ok=True)

import tensorflow as tf  # noqa: E402

from explainit.explainers.minlp_search import MINLSearchExplainer  # noqa: E402
from explainit.priorities.nonlinear import exponential as _nonlinear_exponential  # noqa: E402


logger = logging.getLogger("explainit.development.inspect_minlp")


DATASET_FILES = {
    "communities_crime": "communities_crime_data.pkl",
    "german_credit": "german_credit_data.pkl",
    "credit_card_default": "credit_card_default_data.pkl",
    "lending_club": "lending_club_data.pkl",
}

MODEL_FILES = {
    "communities_crime": "communities_and_crime_model.keras",
    "german_credit": "german_credit_model.keras",
    "credit_card_default": "credit_card_default_model.keras",
    "lending_club": "lending_club_model.keras",
}


# ---------------------------------------------------------------------------
# Self-contained preference builder (mirrors the experiment, no cross-import)
# ---------------------------------------------------------------------------


def create_numerical_preference_function(
    sample_value: float,
    target_value: float,
    dataset_min: float,
    dataset_max: float,
    exemplar_weight: float = 0.5,
    steepness: float = 5,
):
    """Build a piecewise linear+exponential preference for one numerical feature.

    Duplicated here (rather than imported from the experiment) so the
    development directory stays self-contained.
    """

    a = steepness
    midpoint = (sample_value + target_value) / 2.0

    t_inc = np.log(1 + exemplar_weight * (np.exp(a) - 1)) / a
    t_dec = np.log(1 + (1 - exemplar_weight) * (np.exp(a) - 1)) / a

    if sample_value < target_value:
        x0 = midpoint
        x1 = x0 + (target_value - x0) / t_dec
        acceptable_min = sample_value
        acceptable_max = dataset_max
        _sv, _mp = float(sample_value), float(midpoint)
        _x0, _x1, _a = float(x0), float(x1), a

        def preference_func(x):
            x_arr = np.asarray(x, dtype=float)
            scalar_input = x_arr.ndim == 0
            x_arr = np.atleast_1d(x_arr)
            result = np.zeros(x_arr.shape)
            mask_lin = (x_arr >= _sv) & (x_arr <= _mp)
            if np.any(mask_lin):
                denom = _mp - _sv
                if abs(denom) >= 1e-12:
                    t = (x_arr[mask_lin] - _sv) / denom
                    result[mask_lin] = 0.5 + 0.5 * t
                else:
                    result[mask_lin] = 1.0
            mask_exp = x_arr > _mp
            if np.any(mask_exp):
                result[mask_exp] = _nonlinear_exponential(
                    x_arr[mask_exp], x0=_x0, x1=_x1, increasing=False, a=_a
                )
            return float(result.item()) if scalar_input else result
    else:
        x1 = midpoint
        x0 = (target_value - t_inc * x1) / (1 - t_inc)
        acceptable_min = dataset_min
        acceptable_max = sample_value
        _sv, _mp = float(sample_value), float(midpoint)
        _x0, _x1, _a = float(x0), float(x1), a

        def preference_func(x):
            x_arr = np.asarray(x, dtype=float)
            scalar_input = x_arr.ndim == 0
            x_arr = np.atleast_1d(x_arr)
            result = np.zeros(x_arr.shape)
            mask_exp = x_arr < _mp
            if np.any(mask_exp):
                result[mask_exp] = _nonlinear_exponential(
                    x_arr[mask_exp], x0=_x0, x1=_x1, increasing=True, a=_a
                )
            mask_lin = (x_arr >= _mp) & (x_arr <= _sv)
            if np.any(mask_lin):
                denom = _sv - _mp
                if abs(denom) >= 1e-12:
                    t = (x_arr[mask_lin] - _mp) / denom
                    result[mask_lin] = 1.0 - 0.5 * t
                else:
                    result[mask_lin] = 1.0
            return float(result.item()) if scalar_input else result

    acceptable_min = max(dataset_min, min(acceptable_min, dataset_max))
    acceptable_max = max(dataset_min, min(acceptable_max, dataset_max))
    if acceptable_min > acceptable_max:
        acceptable_min = dataset_min
        acceptable_max = dataset_max
    return preference_func, acceptable_min, acceptable_max


def define_preferences(
    sample, target_sample, X_train, feature_names,
    exemplar_weight: float = 0.5, steepness: float = 5,
):
    """Build the per-sample priorities dict used by the experiment."""

    sample = np.array(sample).flatten()
    target_sample = np.array(target_sample).flatten()
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)

    numerical: Dict[int, Dict[str, Any]] = {}
    params_list: List[Dict[str, Any]] = []

    for idx in range(len(sample)):
        sample_val = float(sample[idx])
        target_val = float(target_sample[idx])
        dataset_min = float(X_train_np[:, idx].min())
        dataset_max = float(X_train_np[:, idx].max())
        pref_func, acc_min, acc_max = create_numerical_preference_function(
            sample_value=sample_val, target_value=target_val,
            dataset_min=dataset_min, dataset_max=dataset_max,
            exemplar_weight=exemplar_weight, steepness=steepness,
        )
        numerical[idx] = {
            "function": pref_func,
            "min": acc_min,
            "max": acc_max,
        }
        params_list.append({
            "feature_index": idx,
            "feature_name": (
                feature_names[idx] if idx < len(feature_names) else f"f{idx}"
            ),
            "sample_value": sample_val,
            "target_value": target_val,
            "acceptable_min": acc_min,
            "acceptable_max": acc_max,
            "dataset_min": dataset_min,
            "dataset_max": dataset_max,
        })

    return {"numerical": numerical, "categorical": {}}, params_list


# ---------------------------------------------------------------------------
# Data + model loading (reuses the experiment's pickle/keras caches)
# ---------------------------------------------------------------------------


@dataclass
class DevContext:
    """Container for everything needed to invoke MINLSearchExplainer."""

    dataset_key: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    feature_names: List[str]
    model: Any
    target_sample: Optional[np.ndarray] = None
    config: Dict[str, Any] = field(default_factory=dict)


def _resolve_paths(dataset_key: str) -> Tuple[Path, Path]:
    if dataset_key not in DATASET_FILES:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Supported: {sorted(DATASET_FILES)}"
        )
    data_path = DEV_DATA_DIR / DATASET_FILES[dataset_key]
    model_path = DEV_MODELS_DIR / MODEL_FILES[dataset_key]
    return data_path, model_path


def load_context(
    dataset_key: str,
    *,
    target_probability: float = 0.75,
    target_class: int = 1,
    require_target_sample: bool = True,
) -> DevContext:
    """Load cached dataset + model and choose a default target exemplar."""

    data_path, model_path = _resolve_paths(dataset_key)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cached dataset pickle not found at {data_path}.\n"
            "Populate the development data cache by running:\n"
            "  python -m explainit.development.data_setup "
            f"--source binary_minlp --datasets {dataset_key}\n"
            "or train fresh models with --source train."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained Keras model not found at {model_path}.\n"
            "Populate the development model cache by running:\n"
            "  python -m explainit.development.data_setup "
            f"--source binary_minlp --datasets {dataset_key}"
        )

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    feature_names = data.get("feature_names")
    if hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    feature_names = list(feature_names)

    model = tf.keras.models.load_model(model_path)

    ctx = DevContext(
        dataset_key=dataset_key,
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        feature_names=feature_names,
        model=model,
    )

    if require_target_sample:
        ctx.target_sample = _find_default_target_sample(
            ctx, target_class=target_class, target_probability=target_probability
        )

    return ctx


def _find_default_target_sample(
    ctx: DevContext, *, target_class: int, target_probability: float
) -> Optional[np.ndarray]:
    X = (
        ctx.X_train.values
        if hasattr(ctx.X_train, "values")
        else np.asarray(ctx.X_train)
    )
    y = (
        ctx.y_train.values
        if hasattr(ctx.y_train, "values")
        else np.asarray(ctx.y_train)
    )
    y = np.asarray(y).flatten()
    preds = ctx.model.predict(X, verbose=0).flatten()
    mask = y == target_class
    if not np.any(mask):
        logger.warning(
            "No training samples with class %s; cannot pick target exemplar.",
            target_class,
        )
        return None
    candidate_idx = np.argmin(np.abs(preds[mask] - target_probability))
    target_sample = X[mask][candidate_idx]
    chosen_prob = float(preds[mask][candidate_idx])
    logger.info(
        "Default target exemplar: index=%d of class %d, model prob=%.4f "
        "(requested %.2f).",
        int(np.where(mask)[0][candidate_idx]),
        target_class,
        chosen_prob,
        target_probability,
    )
    return target_sample


def model_predict_fn(model):
    def _predict(X_batch):
        arr = np.asarray(X_batch, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return model.predict(arr, verbose=0).flatten()

    return _predict


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def select_sample_indices(
    ctx: DevContext,
    *,
    sample_index: Optional[int] = None,
    random_samples: Optional[int] = None,
    source_class: int = 0,
    random_seed: int = 42,
) -> List[int]:
    """Pick one or several test-set sample indices to inspect."""

    y_test = (
        ctx.y_test.values
        if hasattr(ctx.y_test, "values")
        else np.asarray(ctx.y_test)
    )
    y_test = np.asarray(y_test).flatten()
    candidate_pool = np.where(y_test == source_class)[0]

    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(y_test):
            raise IndexError(
                f"Sample index {sample_index} is out of bounds "
                f"(test set size {len(y_test)})."
            )
        return [int(sample_index)]

    if random_samples is None or random_samples <= 0:
        random_samples = 1

    if len(candidate_pool) == 0:
        raise ValueError(
            f"No test samples with class {source_class} available."
        )

    rng = np.random.default_rng(random_seed)
    take = min(random_samples, len(candidate_pool))
    return [int(i) for i in rng.choice(candidate_pool, size=take, replace=False)]


def build_sample_preferences(
    ctx: DevContext,
    sample: np.ndarray,
    *,
    exemplar_weight: float = 0.85,
    steepness: float = 5.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the priorities dict expected by MINLSearchExplainer."""

    if ctx.target_sample is None:
        raise RuntimeError(
            "DevContext has no target_sample; call load_context() with "
            "require_target_sample=True or set ctx.target_sample manually."
        )
    return define_preferences(
        sample=sample,
        target_sample=ctx.target_sample,
        X_train=ctx.X_train,
        feature_names=ctx.feature_names,
        exemplar_weight=exemplar_weight,
        steepness=steepness,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _bar(title: str, char: str = "=") -> None:
    logger.info(char * 78)
    logger.info(title)
    logger.info(char * 78)


def _format_change(name: str, before: float, after: float) -> str:
    delta = after - before
    arrow = "->" if abs(delta) > 1e-9 else "=="
    return f"  {name:<24s} {before:>12.4f} {arrow} {after:>12.4f}  (Δ={delta:+.4f})"


def describe_priorities(
    priorities: Dict[str, Any],
    feature_names: Sequence[str],
    sample: np.ndarray,
    *,
    head: int = 10,
) -> None:
    _bar("PRIORITIES PREVIEW (first {} features)".format(head), char="-")
    numerical = priorities.get("numerical", {})
    shown = 0
    for idx, cfg in numerical.items():
        if shown >= head:
            logger.info("  ... %d more numerical priorities not shown ...",
                        len(numerical) - head)
            break
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        func = cfg.get("function") if isinstance(cfg, dict) else None
        if func is None:
            logger.info(
                "  NUM %-24s idx=%2d | non-actionable, fixed=[%.4f, %.4f]",
                name, idx, float(cfg["min"]), float(cfg["max"]),
            )
        else:
            weight_at_sample = float(np.asarray(func(float(sample[idx]))).squeeze())
            logger.info(
                "  NUM %-24s idx=%2d | sample=%.4f | bounds=[%.4f, %.4f] | "
                "pref(sample)=%.4f",
                name, idx, float(sample[idx]),
                float(cfg["min"]), float(cfg["max"]),
                weight_at_sample,
            )
        shown += 1


def describe_shapley(explainer: MINLSearchExplainer, feature_names: Sequence[str]) -> None:
    _bar("SHAPLEY VALUES", char="-")
    shap = explainer.sample_state.shapley_values or {}
    num = shap.get("numerical", {}) or {}
    cat = shap.get("categorical", {}) or {}
    items = sorted(num.items(), key=lambda kv: -abs(kv[1]))
    for idx, val in items[:10]:
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        logger.info("  NUM phi[%-24s idx=%2d] = %+.6f", name, idx, float(val))
    if len(items) > 10:
        logger.info("  ... %d more numerical phi values not shown ...",
                    len(items) - 10)
    for group, val in cat.items():
        group_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                       for i in group]
        logger.info("  CAT phi[%s] = %+.6f", group_names, float(val))


def evaluate_cf(
    ctx: DevContext,
    sample: np.ndarray,
    cf: np.ndarray,
    *,
    threshold: float,
    target_class: int,
) -> Dict[str, Any]:
    sample = np.asarray(sample, dtype=float).flatten()
    cf = np.asarray(cf, dtype=float).flatten()
    pred = float(ctx.model.predict(cf.reshape(1, -1), verbose=0)[0, 0])
    pred_class = int(pred >= threshold)
    delta = cf - sample
    l1 = float(np.sum(np.abs(delta)))
    l2 = float(np.linalg.norm(delta))
    changed = [
        i for i in range(len(sample)) if abs(delta[i]) > 1e-6
    ]
    return {
        "prediction": pred,
        "predicted_class": pred_class,
        "target_achieved": pred_class == target_class,
        "l1_distance": l1,
        "l2_distance": l2,
        "sparsity_changed": changed,
        "n_changed": len(changed),
        "delta": delta,
    }


def describe_cf(
    feature_names: Sequence[str],
    sample: np.ndarray,
    cf: np.ndarray,
    metrics: Dict[str, Any],
    *,
    rank: int,
    threshold: float,
    target_class: int,
    max_features: int = 12,
) -> None:
    _bar(
        f"COUNTERFACTUAL #{rank}  pred={metrics['prediction']:.4f} "
        f"class={metrics['predicted_class']} "
        f"valid={metrics['target_achieved']}",
        char="-",
    )
    logger.info(
        "  threshold=%.2f target_class=%d L1=%.4f L2=%.4f "
        "n_features_changed=%d",
        threshold, target_class,
        metrics["l1_distance"], metrics["l2_distance"], metrics["n_changed"],
    )
    if not metrics["target_achieved"]:
        gap = metrics["prediction"] - threshold
        logger.warning(
            "  Validity FAIL: model probability %.4f did not cross threshold "
            "%.4f for class %d (gap=%+.4f). The Shapley linear approximation "
            "likely over-estimated the model's response on this combination.",
            metrics["prediction"], threshold, target_class, gap,
        )
    changed = metrics["sparsity_changed"][:max_features]
    if not changed:
        logger.info("  No feature actually changed (CF == original).")
        return
    logger.info("  Top changed features:")
    for idx in changed:
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        logger.info(_format_change(name, sample[idx], cf[idx]))
    if metrics["n_changed"] > max_features:
        logger.info(
            "  ... %d more changed features not shown ...",
            metrics["n_changed"] - max_features,
        )


# ---------------------------------------------------------------------------
# Main inspection routine
# ---------------------------------------------------------------------------


@dataclass
class InspectionResult:
    sample_index: int
    sample: np.ndarray
    original_prediction: float
    counterfactuals: List[np.ndarray] = field(default_factory=list)
    counterfactual_metrics: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def n_valid(self) -> int:
        return sum(1 for m in self.counterfactual_metrics if m["target_achieved"])

    @property
    def validity_rate(self) -> float:
        if not self.counterfactual_metrics:
            return 0.0
        return self.n_valid / len(self.counterfactual_metrics)


def inspect_sample(
    ctx: DevContext,
    sample_index: int,
    *,
    num_cfs: int = 2,
    target_class: int = 1,
    target_probability: float = 0.75,
    threshold: float = 0.5,
    exemplar_weight: float = 0.85,
    steepness: float = 5.0,
    target_exemplar_epsilon: float = 0.05,
    epsilon: float = 0.1,
    max_iterations: int = 200,
    shap_approx: bool = True,
    shap_num_samples: int = 200,
) -> InspectionResult:
    """Run MINLSearchExplainer on a single sample with verbose diagnostics."""

    X_test = (
        ctx.X_test.values
        if hasattr(ctx.X_test, "values")
        else np.asarray(ctx.X_test)
    )
    sample = X_test[sample_index].astype(float)
    original_pred = float(ctx.model.predict(sample.reshape(1, -1), verbose=0)[0, 0])
    original_class = int(original_pred >= threshold)

    _bar(
        f"DATASET={ctx.dataset_key.upper()} SAMPLE_INDEX={sample_index} "
        f"original_pred={original_pred:.4f} original_class={original_class}"
    )

    result = InspectionResult(
        sample_index=sample_index,
        sample=sample.copy(),
        original_prediction=original_pred,
    )

    try:
        preferences, _ = build_sample_preferences(
            ctx, sample,
            exemplar_weight=exemplar_weight,
            steepness=steepness,
        )
    except Exception as exc:
        logger.exception("Failed to build per-sample priorities: %s", exc)
        result.error = f"priorities: {exc}"
        return result

    describe_priorities(preferences, ctx.feature_names, sample)

    X_train = (
        ctx.X_train.values
        if hasattr(ctx.X_train, "values")
        else np.asarray(ctx.X_train)
    )

    explainer = MINLSearchExplainer(
        model_pred=model_predict_fn(ctx.model),
        priorities=preferences,
        sample=sample.tolist(),
        target=float(target_probability),
        dataset=X_train.copy(),
        target_exemplar_epsilon=float(target_exemplar_epsilon),
        epsilon=float(epsilon),
    )

    started = time.perf_counter()
    try:
        cfs, predictions, scores, found_counts = (
            explainer.find_counterfactuals_for_binary(
                target_class=int(target_class),
                threshold=float(threshold),
                expected_counterfactuals=int(num_cfs),
                max_iterations=int(max_iterations),
                shap_approx=bool(shap_approx),
                num_samples=int(shap_num_samples),
                return_top_n=int(num_cfs),
            )
        )
    except Exception as exc:
        logger.exception("MINLP search failed: %s", exc)
        result.error = f"search: {exc}"
        result.elapsed_seconds = time.perf_counter() - started
        return result
    result.elapsed_seconds = time.perf_counter() - started

    # Best-effort: dump shapley values after they are computed.
    try:
        describe_shapley(explainer, ctx.feature_names)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Could not describe Shapley values: %s", exc)

    if not cfs:
        logger.warning(
            "No counterfactuals returned. Try relaxing bounds, raising "
            "max_iterations, or switching shap_approx off."
        )
        return result

    for rank, (cf, pred, score, iters) in enumerate(
        zip(cfs, predictions, scores, found_counts), start=1
    ):
        cf_arr = np.asarray(cf, dtype=float).flatten()
        metrics = evaluate_cf(
            ctx, sample, cf_arr,
            threshold=threshold, target_class=target_class,
        )
        metrics["explainer_prediction"] = float(pred)
        metrics["preference_score_internal"] = float(score)
        metrics["found_iteration"] = int(iters)
        describe_cf(
            ctx.feature_names, sample, cf_arr, metrics,
            rank=rank, threshold=threshold, target_class=target_class,
        )
        result.counterfactuals.append(cf_arr)
        result.counterfactual_metrics.append(metrics)

    _bar(
        f"SUMMARY sample={sample_index}: valid {result.n_valid}/"
        f"{len(result.counterfactual_metrics)} "
        f"({result.validity_rate*100:.1f}%) "
        f"elapsed={result.elapsed_seconds:.2f}s",
        char="-",
    )
    return result


def inspect_samples(
    ctx: DevContext,
    sample_indices: Sequence[int],
    **kwargs: Any,
) -> List[InspectionResult]:
    results: List[InspectionResult] = []
    for idx in sample_indices:
        results.append(inspect_sample(ctx, idx, **kwargs))
    return results


def summarise_runs(results: Sequence[InspectionResult]) -> None:
    if not results:
        logger.info("No inspection results to summarise.")
        return
    _bar("OVERALL SUMMARY")
    total_cfs = sum(len(r.counterfactual_metrics) for r in results)
    total_valid = sum(r.n_valid for r in results)
    elapsed = sum(r.elapsed_seconds for r in results)
    rate = (total_valid / total_cfs * 100.0) if total_cfs else 0.0
    logger.info(
        "Inspected %d sample(s); produced %d CF(s); %d valid (%.1f%%); "
        "total elapsed %.2fs.",
        len(results), total_cfs, total_valid, rate, elapsed,
    )
    for r in results:
        rate_i = r.validity_rate * 100.0
        logger.info(
            "  sample %4d | orig pred=%.4f | %d/%d valid (%.1f%%) | "
            "elapsed=%.2fs%s",
            r.sample_index, r.original_prediction, r.n_valid,
            len(r.counterfactual_metrics), rate_i, r.elapsed_seconds,
            f" | error={r.error}" if r.error else "",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        description=(
            "Investigate MINLSearchExplainer on the binary_minlp experiment "
            "datasets and pre-trained models."
        )
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=sorted(DATASET_FILES),
        default="german_credit",
        help="Dataset whose cached pickle + Keras model should be loaded.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--sample-index", "-i", type=int, default=None,
        help="Index into the cached test set to inspect (default: random).",
    )
    selector.add_argument(
        "--random-samples", "-N", type=int, default=None,
        help="Randomly pick this many source-class test samples to inspect.",
    )
    parser.add_argument(
        "--source-class", type=int, default=0,
        help="Class label of the samples we want to flip (default: 0).",
    )
    parser.add_argument(
        "--target-class", type=int, default=1,
        help="Desired counterfactual class (default: 1).",
    )
    parser.add_argument(
        "--num-cfs", "-k", type=int, default=2,
        help="Number of counterfactuals to ask for per sample.",
    )
    parser.add_argument(
        "--target-probability", type=float, default=0.75,
        help="Reference probability used to find a target exemplar.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binary-classification decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--exemplar-weight", type=float, default=0.85,
        help="exemplar_weight for per-sample preference functions.",
    )
    parser.add_argument(
        "--steepness", type=float, default=5.0,
        help="Steepness parameter for the exponential preference.",
    )
    parser.add_argument(
        "--target-exemplar-epsilon", type=float, default=0.05,
        help="Epsilon for the target exemplar search inside the explainer.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Tolerance used by the Shapley-based constraint inside MINLP.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=200,
        help="Max SLSQP iterations per categorical combination.",
    )
    parser.add_argument(
        "--shap-approx", action="store_true", default=True,
        help="Use Monte-Carlo Shapley approximation (default).",
    )
    parser.add_argument(
        "--shap-exact", dest="shap_approx", action="store_false",
        help="Disable Shapley approximation (slow for high-dim datasets).",
    )
    parser.add_argument(
        "--shap-num-samples", type=int, default=200,
        help="Sample count for Shapley approximation.",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Seed used when picking random samples.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    ctx = load_context(
        args.dataset,
        target_probability=args.target_probability,
        target_class=args.target_class,
    )

    sample_indices = select_sample_indices(
        ctx,
        sample_index=args.sample_index,
        random_samples=args.random_samples,
        source_class=args.source_class,
        random_seed=args.random_seed,
    )

    results = inspect_samples(
        ctx,
        sample_indices,
        num_cfs=args.num_cfs,
        target_class=args.target_class,
        target_probability=args.target_probability,
        threshold=args.threshold,
        exemplar_weight=args.exemplar_weight,
        steepness=args.steepness,
        target_exemplar_epsilon=args.target_exemplar_epsilon,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        shap_approx=args.shap_approx,
        shap_num_samples=args.shap_num_samples,
    )
    summarise_runs(results)


if __name__ == "__main__":
    main()
