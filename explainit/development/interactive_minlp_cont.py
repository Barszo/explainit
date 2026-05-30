"""Per-sample MINLP workbench for **continuous-target** (regression) models.

Sibling of ``interactive_minlp.py``: same idea, but works against regression
models (continuous y) and uses ``MINLSearchExplainer.find_counterfactuals``
(the original continuous flow) instead of the binary helper.

Datasets shipped out of the box (no manual download required beyond what
sklearn caches once in ``~/scikit_learn_data``):

* ``diabetes``           — sklearn bundled, 10 continuous features,
                            target = quantitative disease-progression score.
* ``california_housing`` — sklearn ``fetch_california_housing``, 8 features,
                            target = median house value (~$100k units).
* ``synthetic``          — in-memory toy dataset generated from a known
                            linear combination + noise; instant to load and
                            useful as a controlled baseline.

The training pipeline standard-scales the features and **MinMax-scales the
target into [0, 1]** so ``target`` and ``epsilon`` for MINLP have a
predictable, dataset-independent interpretation.

Two ways to use this file:

1. **Edit-and-run script** — change the ``USER_*`` constants and
   ``build_my_priorities`` near the bottom and run::

       python -m explainit.development.interactive_minlp_cont

2. **Library use** in a REPL / notebook::

       from explainit.development.interactive_minlp_cont import (
           load_cont_context, show_features, show_sample,
           find_continuous_target_exemplar, PriorityBuilder,
           run_minlp_cont_on_sample,
       )
       ctx = load_cont_context("diabetes")
       show_features(ctx)
       target_y = 0.75
       exemplar = find_continuous_target_exemplar(ctx, target_y=target_y)
       sample = ctx.X_test.iloc[0].to_numpy(dtype=float)
       priorities = (
           PriorityBuilder(ctx, sample=sample)
               .add_exponential("bmi",  x0=sample[2], x1=exemplar[2],
                                increasing=exemplar[2] > sample[2])
               .add_linear("bp",        x0=sample[3], x1=exemplar[3],
                           increasing=exemplar[3] > sample[3])
               .build()
       )
       run_minlp_cont_on_sample(ctx, 0, priorities, target_y=target_y,
                                epsilon=0.05)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import pickle
import re
import ssl
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEV_DIR = Path(__file__).resolve().parent
DATA_CONT_DIR = DEV_DIR / "data_cont"
MODELS_CONT_DIR = DEV_DIR / "models_cont"
IMAGES_DIR = DEV_DIR / "images"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_CONT_DIR.mkdir(exist_ok=True)
MODELS_CONT_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

import tensorflow as tf  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402

from explainit.explainers.minlp_search import MINLSearchExplainer  # noqa: E402
from explainit.development.inspect_minlp import (  # noqa: E402
    DevContext,
    describe_priorities,
    describe_shapley,
    model_predict_fn,
)
from explainit.development.interactive_minlp import (  # noqa: E402
    PriorityBuilder,
    _as_array,
)
from explainit.utils.priority_plots import plot_priorities  # noqa: E402
from explainit.utils.dataset_analyzer import analyze_dataset  # noqa: E402


logger = logging.getLogger("explainit.development.interactive_minlp_cont")
workflow_logger = logging.getLogger("explainit.workflow")


CONT_DATASETS = ("diabetes", "california_housing", "synthetic")

DATASET_FILE_TEMPLATE = "{key}_cont_data.pkl"
MODEL_FILE_TEMPLATE = "{key}_cont_model.keras"


_TARGET_NAME_BY_DATASET = {
    "diabetes": "disease_progression",
    "california_housing": "median_house_value",
    "synthetic": "synthetic_target",
}


# ---------------------------------------------------------------------------
# Continuous-target DevContext (extends the binary one with y scaling info)
# ---------------------------------------------------------------------------


@dataclass
class ContDevContext(DevContext):
    """Same fields as ``DevContext`` plus continuous-target metadata.

    ``y_scaler`` is the MinMaxScaler used to map the raw target into [0, 1];
    keep it so you can invert predictions back to the original units when
    reporting results.
    """

    y_scaler: Optional[MinMaxScaler] = None
    raw_target_min: float = 0.0
    raw_target_max: float = 1.0


# ---------------------------------------------------------------------------
# Dataset loaders (no external downloads beyond what sklearn caches)
# ---------------------------------------------------------------------------


def _load_diabetes_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from sklearn.datasets import load_diabetes
    bunch = load_diabetes(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.astype(float).to_numpy()
    feature_names = list(bunch.feature_names)
    return X, y, feature_names


def _load_california_housing_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from sklearn.datasets import fetch_california_housing
    # macOS Python sometimes ships expired root certs; allow fallback.
    try:
        bunch = fetch_california_housing(as_frame=True)
    except Exception as exc:
        logger.warning("fetch_california_housing failed (%s); retrying with SSL disabled.", exc)
        ssl._create_default_https_context = ssl._create_unverified_context
        bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.astype(float).to_numpy()
    feature_names = list(bunch.feature_names)
    return X, y, feature_names


def _load_synthetic_raw(
    n_samples: int = 2000,
    n_features: int = 6,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    X_np = rng.normal(size=(n_samples, n_features))
    weights = np.linspace(1.0, 0.2, n_features)
    bias = 0.3
    y = X_np @ weights + bias + rng.normal(scale=noise, size=n_samples)
    feature_names = [f"x{i}" for i in range(n_features)]
    X = pd.DataFrame(X_np, columns=feature_names)
    return X, y.astype(float), feature_names


def _build_regression_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def _prepare_and_train(
    key: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 60,
    batch_size: int = 32,
) -> Tuple[Dict[str, Any], tf.keras.Model]:
    if key == "diabetes":
        X_raw, y_raw, feature_names = _load_diabetes_raw()
    elif key == "california_housing":
        X_raw, y_raw, feature_names = _load_california_housing_raw()
    elif key == "synthetic":
        X_raw, y_raw, feature_names = _load_synthetic_raw()
    else:
        raise ValueError(f"Unknown dataset '{key}'. Supported: {CONT_DATASETS}")

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state,
    )

    x_scaler = StandardScaler()
    X_train = pd.DataFrame(
        x_scaler.fit_transform(X_train_raw),
        columns=feature_names, index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        x_scaler.transform(X_test_raw),
        columns=feature_names, index=X_test_raw.index,
    )

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    model = _build_regression_model(X_train.shape[1])
    model.fit(
        X_train.values, y_train,
        validation_data=(X_test.values, y_test),
        epochs=epochs, batch_size=batch_size, verbose=0,
    )

    train_loss = float(model.evaluate(X_train.values, y_train, verbose=0))
    test_loss = float(model.evaluate(X_test.values, y_test, verbose=0))
    logger.info(
        "Trained %s regression model | MSE train=%.4f test=%.4f (scaled y in [0,1])",
        key, train_loss, test_loss,
    )

    data_dict = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "raw_target_min": float(np.min(y_raw)),
        "raw_target_max": float(np.max(y_raw)),
    }
    return data_dict, model


def setup_continuous_dataset(
    key: str,
    *,
    force: bool = False,
    epochs: int = 60,
    batch_size: int = 32,
) -> Tuple[Path, Path]:
    """Cache dataset pickle + trained regression Keras model for ``key``."""

    if key not in CONT_DATASETS:
        raise ValueError(f"Unknown dataset '{key}'. Supported: {CONT_DATASETS}")

    data_path = DATA_CONT_DIR / DATASET_FILE_TEMPLATE.format(key=key)
    model_path = MODELS_CONT_DIR / MODEL_FILE_TEMPLATE.format(key=key)

    if data_path.exists() and model_path.exists() and not force:
        logger.info("Cache present for '%s' (use force=True to retrain).", key)
        return data_path, model_path

    logger.info("Preparing '%s' (epochs=%d, batch=%d)...", key, epochs, batch_size)
    data_dict, model = _prepare_and_train(
        key, epochs=epochs, batch_size=batch_size,
    )
    with open(data_path, "wb") as handle:
        pickle.dump(data_dict, handle)
    model.save(model_path)
    logger.info("Saved %s and %s", data_path, model_path)
    return data_path, model_path


# ---------------------------------------------------------------------------
# Loading + display helpers
# ---------------------------------------------------------------------------


def load_cont_context(
    key: str,
    *,
    force_setup: bool = False,
    epochs: int = 60,
    batch_size: int = 32,
) -> ContDevContext:
    """Load (and if needed, prepare) a continuous-target dev context."""

    data_path, model_path = setup_continuous_dataset(
        key, force=force_setup, epochs=epochs, batch_size=batch_size,
    )
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    model = tf.keras.models.load_model(model_path)

    feature_names = list(data["feature_names"])
    return ContDevContext(
        dataset_key=key,
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        feature_names=feature_names,
        model=model,
        y_scaler=data.get("y_scaler"),
        raw_target_min=float(data.get("raw_target_min", 0.0)),
        raw_target_max=float(data.get("raw_target_max", 1.0)),
    )


def show_features(ctx: ContDevContext) -> None:
    logger.info(
        "Features in dataset '%s' (%d total):",
        ctx.dataset_key, len(ctx.feature_names),
    )
    for idx, name in enumerate(ctx.feature_names):
        logger.info("  %3d  %s", idx, name)


def _unscale_y(ctx: ContDevContext, y_scaled: float) -> float:
    if ctx.y_scaler is None:
        return float(y_scaled)
    arr = np.asarray([[float(y_scaled)]])
    return float(ctx.y_scaler.inverse_transform(arr)[0, 0])


def show_sample(ctx: ContDevContext, sample_index: int, max_features: int = 25) -> np.ndarray:
    X_test = _as_array(ctx.X_test)
    if sample_index < 0 or sample_index >= len(X_test):
        raise IndexError(
            f"sample_index {sample_index} out of range (test size {len(X_test)})"
        )
    sample = X_test[sample_index].astype(float)
    pred_scaled = float(ctx.model.predict(sample.reshape(1, -1), verbose=0)[0, 0])
    y_test_scaled = float(np.asarray(ctx.y_test).flatten()[sample_index])
    logger.info(
        "Sample %d | true y(scaled)=%.4f (raw=%.4f) | model y(scaled)=%.4f (raw=%.4f)",
        sample_index, y_test_scaled, _unscale_y(ctx, y_test_scaled),
        pred_scaled, _unscale_y(ctx, pred_scaled),
    )
    for idx in range(min(len(sample), max_features)):
        logger.info("  %3d  %-24s = %.4f", idx, ctx.feature_names[idx], sample[idx])
    if len(sample) > max_features:
        logger.info("  ... %d more features not shown ...", len(sample) - max_features)
    return sample


def run_dataset_analysis(
    ctx: ContDevContext,
    *,
    output_dir: Optional[Path] = None,
    sample: Optional[np.ndarray] = None,
    exemplar: Optional[np.ndarray] = None,
    use_train: bool = True,
) -> Path:
    """Run :func:`analyze_dataset` on the loaded context and return the folder.

    By default the analysis covers the training split (it is larger and
    therefore yields more reliable distribution estimates). Plots and the
    textual ``summary.txt`` are written under
    ``development/images/<dataset_key>_analysis/``.
    """

    target_dir = Path(output_dir) if output_dir is not None else (
        IMAGES_DIR / f"{ctx.dataset_key}_analysis"
    )
    if use_train:
        X = ctx.X_train
        y = ctx.y_train
    else:
        X = ctx.X_test
        y = ctx.y_test

    sample_target_value: Optional[float] = None
    exemplar_target_value: Optional[float] = None
    if sample is not None:
        try:
            sample_target_value = float(
                ctx.model.predict(np.asarray(sample, dtype=float).reshape(1, -1), verbose=0)[0, 0]
            )
        except Exception as exc:
            logger.warning("Could not compute sample target value for analysis: %s", exc)
    if exemplar is not None:
        try:
            exemplar_target_value = float(
                ctx.model.predict(np.asarray(exemplar, dtype=float).reshape(1, -1), verbose=0)[0, 0]
            )
        except Exception as exc:
            logger.warning("Could not compute exemplar target value for analysis: %s", exc)

    logger.info("Running dataset analysis on '%s' -> %s", ctx.dataset_key, target_dir)
    report = analyze_dataset(
        X=X,
        y=y,
        feature_names=ctx.feature_names,
        target_name=_TARGET_NAME_BY_DATASET.get(ctx.dataset_key, "target"),
        dataset_key=ctx.dataset_key,
        output_dir=target_dir,
        sample=sample,
        exemplar=exemplar,
        sample_target_value=sample_target_value,
        exemplar_target_value=exemplar_target_value,
    )
    logger.info(
        "Dataset analysis: %d feature(s), %d plot(s), summary at %s",
        len(report.features), len(report.saved_plots), report.summary_text_path,
    )
    return target_dir


def find_continuous_target_exemplar(
    ctx: ContDevContext,
    *,
    target_y: float,
) -> np.ndarray:
    """Return a training-set point whose model prediction is close to ``target_y``.

    ``target_y`` is interpreted on the *scaled* (model output) space, which
    is [0, 1] for the bundled datasets.
    """

    X = _as_array(ctx.X_train)
    preds = ctx.model.predict(X, verbose=0).flatten()
    idx = int(np.argmin(np.abs(preds - float(target_y))))
    exemplar = X[idx].astype(float)
    logger.info(
        "Picked continuous target exemplar idx=%d | model y(scaled)=%.4f (raw=%.4f) "
        "(requested %.4f).",
        idx, float(preds[idx]), _unscale_y(ctx, float(preds[idx])), target_y,
    )
    return exemplar


# ---------------------------------------------------------------------------
# Continuous CF evaluation
# ---------------------------------------------------------------------------


def evaluate_cont_cf(
    ctx: ContDevContext,
    sample: np.ndarray,
    cf: np.ndarray,
    *,
    target_y: float,
    epsilon: float,
) -> Dict[str, Any]:
    sample = np.asarray(sample, dtype=float).flatten()
    cf = np.asarray(cf, dtype=float).flatten()
    pred_scaled = float(ctx.model.predict(cf.reshape(1, -1), verbose=0)[0, 0])
    delta = cf - sample
    l1 = float(np.sum(np.abs(delta)))
    l2 = float(np.linalg.norm(delta))
    changed = [i for i in range(len(sample)) if abs(delta[i]) > 1e-6]
    gap = pred_scaled - float(target_y)
    return {
        "prediction_scaled": pred_scaled,
        "prediction_raw": _unscale_y(ctx, pred_scaled),
        "target_scaled": float(target_y),
        "target_raw": _unscale_y(ctx, float(target_y)),
        "gap": gap,
        "within_epsilon": abs(gap) <= float(epsilon),
        "l1_distance": l1,
        "l2_distance": l2,
        "sparsity_changed": changed,
        "n_changed": len(changed),
        "delta": delta,
    }


def describe_cont_cf(
    feature_names: Sequence[str],
    sample: np.ndarray,
    cf: np.ndarray,
    metrics: Dict[str, Any],
    *,
    epsilon: float,
    max_features: int = 12,
) -> None:
    logger.info("-" * 78)
    logger.info(
        "COUNTERFACTUAL | y(scaled)=%.4f (raw=%.4f) | target(scaled)=%.4f "
        "(raw=%.4f) | gap=%+.4f | within ±%.4f: %s",
        metrics["prediction_scaled"], metrics["prediction_raw"],
        metrics["target_scaled"], metrics["target_raw"],
        metrics["gap"], epsilon, metrics["within_epsilon"],
    )
    logger.info(
        "  L1=%.4f L2=%.4f n_features_changed=%d",
        metrics["l1_distance"], metrics["l2_distance"], metrics["n_changed"],
    )
    if not metrics["within_epsilon"]:
        logger.warning(
            "  Validity FAIL: model output %.4f is outside ±%.4f of target %.4f. "
            "Consider relaxing `epsilon`, tightening `target_exemplar_epsilon`, "
            "or revisiting Shapley assumptions (linear approximation may misfire).",
            metrics["prediction_scaled"], epsilon, metrics["target_scaled"],
        )
    changed = metrics["sparsity_changed"][:max_features]
    if not changed:
        logger.info("  CF identical to original (no feature changed).")
        return
    logger.info("  Top changed features:")
    for idx in changed:
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        b = float(sample[idx]); a = float(cf[idx])
        logger.info(
            "  %-24s %+12.4f -> %+12.4f  (Δ=%+.4f)",
            name, b, a, a - b,
        )
    if metrics["n_changed"] > max_features:
        logger.info(
            "  ... %d more changed features not shown ...",
            metrics["n_changed"] - max_features,
        )


# ---------------------------------------------------------------------------
# Per-sample MINLP runner (continuous)
# ---------------------------------------------------------------------------


def save_priority_plots(
    ctx: ContDevContext,
    sample: np.ndarray,
    priorities: Dict[str, Any],
    *,
    sample_index: Optional[int] = None,
    exemplar: Optional[np.ndarray] = None,
    target_threshold: Optional[float] = None,
    output_dir: Path = IMAGES_DIR,
) -> List[Path]:
    """Render priority plots for ``priorities`` into ``output_dir``.

    The directory defaults to ``development/images/`` and a per-sample
    sub-folder is used when ``sample_index`` is supplied so re-runs do not
    overwrite each other.
    """

    target_dir = Path(output_dir)
    if sample_index is not None:
        target_dir = target_dir / f"{ctx.dataset_key}_sample_{sample_index}"
    target_dir.mkdir(parents=True, exist_ok=True)

    written = plot_priorities(
        priorities,
        sample=sample,
        exemplar=exemplar,
        feature_matrix=_as_array(ctx.X_train),
        target_values=np.asarray(ctx.y_train, dtype=float).flatten(),
        target_name=_TARGET_NAME_BY_DATASET.get(ctx.dataset_key, "target"),
        target_threshold=target_threshold,
        feature_names=ctx.feature_names,
        save_dir=target_dir,
        show=False,
    )
    if written:
        logger.info("Saved %d priority plot(s) to %s", len(written), target_dir)
    else:
        logger.info("No actionable priorities to plot (nothing written to %s).",
                    target_dir)
    return written


def run_minlp_cont_on_sample(
    ctx: ContDevContext,
    sample_index: int,
    priorities: Dict[str, Any],
    *,
    target_y: float,
    epsilon: float = 0.05,
    target_exemplar_epsilon: float = 0.10,
    shap_approx: bool = True,
    shap_num_samples: int = 200,
    save_priority_plots_to: Optional[Path] = IMAGES_DIR,
) -> Dict[str, Any]:
    """Invoke ``MINLSearchExplainer.find_counterfactuals`` on one sample.

    Returns a dict with the original sample, the single CF returned (the
    explainer's continuous flow returns the best one across categorical
    combinations), and a metrics breakdown.
    """

    X_test = _as_array(ctx.X_test)
    sample = X_test[sample_index].astype(float)
    original_pred = float(ctx.model.predict(sample.reshape(1, -1), verbose=0)[0, 0])

    logger.info(
        "Running continuous MINLP on dataset=%s sample=%d "
        "(orig y(scaled)=%.4f raw=%.4f, target y(scaled)=%.4f raw=%.4f, "
        "epsilon=%.4f)",
        ctx.dataset_key, sample_index,
        original_pred, _unscale_y(ctx, original_pred),
        float(target_y), _unscale_y(ctx, float(target_y)),
        epsilon,
    )

    priority_exemplar: Optional[np.ndarray] = None
    try:
        priority_exemplar = find_continuous_target_exemplar(
            ctx, target_y=float(target_y)
        )
    except Exception as exc:
        logger.warning("Could not compute exemplar for priority plots: %s", exc)

    describe_priorities(priorities, ctx.feature_names, sample)

    if save_priority_plots_to is not None:
        try:
            save_priority_plots(
                ctx, sample, priorities,
                sample_index=sample_index,
                exemplar=priority_exemplar,
                target_threshold=float(target_y),
                output_dir=save_priority_plots_to,
            )
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            logger.warning("Failed to save priority plots: %s", exc)

    X_train_np = _as_array(ctx.X_train)
    explainer = MINLSearchExplainer(
        model_pred=model_predict_fn(ctx.model),
        priorities=priorities,
        sample=sample.tolist(),
        target=float(target_y),
        dataset=X_train_np.copy(),
        target_exemplar_epsilon=float(target_exemplar_epsilon),
        epsilon=float(epsilon),
        workflow_logger=workflow_logger,
        feature_names=ctx.feature_names,
    )

    started = time.perf_counter()
    cf_arr: Optional[np.ndarray] = None
    error: Optional[str] = None
    try:
        cf_raw = explainer.find_counterfactuals(
            shap_approx=bool(shap_approx),
            num_samples=int(shap_num_samples),
        )
        cf_arr = np.asarray(cf_raw, dtype=float).flatten()
    except Exception as exc:
        logger.exception("Continuous MINLP search failed: %s", exc)
        error = str(exc)
    elapsed = time.perf_counter() - started

    try:
        describe_shapley(explainer, ctx.feature_names)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Could not describe Shapley values: %s", exc)

    metrics: Optional[Dict[str, Any]] = None
    if cf_arr is not None:
        metrics = evaluate_cont_cf(
            ctx, sample, cf_arr, target_y=target_y, epsilon=epsilon,
        )
        describe_cont_cf(
            ctx.feature_names, sample, cf_arr, metrics, epsilon=epsilon,
        )

    logger.info(
        "Sample %d done | within_eps=%s | elapsed=%.2fs%s",
        sample_index,
        "n/a" if metrics is None else metrics["within_epsilon"],
        elapsed,
        f" | error={error}" if error else "",
    )

    return {
        "sample_index": sample_index,
        "sample": sample,
        "original_prediction_scaled": original_pred,
        "original_prediction_raw": _unscale_y(ctx, original_pred),
        "counterfactual": cf_arr,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "error": error,
    }


def run_minlp_cont_on_samples(
    ctx: ContDevContext,
    sample_indices: Iterable[int],
    priorities_builder: Callable[[ContDevContext, np.ndarray], Dict[str, Any]],
    *,
    target_y: float,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    X_test = _as_array(ctx.X_test)
    results: List[Dict[str, Any]] = []
    for idx in sample_indices:
        sample = X_test[int(idx)].astype(float)
        priorities = priorities_builder(ctx, sample)
        results.append(
            run_minlp_cont_on_sample(
                ctx, int(idx), priorities,
                target_y=target_y, **kwargs,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Editable template — adjust USER_* and build_my_priorities for your case
# ---------------------------------------------------------------------------


USER_DATASET = "diabetes"
USER_SAMPLE_INDICES: Sequence[int] = (0,)
USER_TARGET_Y_SCALED: float = 0.75
USER_EPSILON: float = 0.05
USER_TARGET_EXEMPLAR_EPSILON: float = 0.10
USER_SHAP_APPROX = True
USER_SHAP_NUM_SAMPLES = 200


def build_my_priorities(ctx: ContDevContext, sample: np.ndarray) -> Dict[str, Any]:
    """Default example: nudge every feature toward the target exemplar.

    Replace freely. The PriorityBuilder is shared with the binary workbench;
    you can mix ``add_linear``, ``add_exponential``, ``add_constant_pref``,
    ``set_non_actionable``, ``add_categorical`` (the last requires the
    feature indices form a contiguous block from the leading numerical
    features — see the comment in ``examples/continuous_minlp_search_example.py``).
    """

    target = find_continuous_target_exemplar(ctx, target_y=USER_TARGET_Y_SCALED)
    pb = PriorityBuilder(ctx, sample=sample)
    for idx, name in enumerate(ctx.feature_names):
        x0 = float(sample[idx])
        x1 = float(target[idx])
        if abs(x1 - x0) < 1e-9:
            pb.add_constant_pref(idx, weight=0.5)
        else:
            pb.add_exponential(
                idx, x0=x0, x1=x1, increasing=(x1 > x0), a=5.0,
            )
    return pb.build()


class _ExecutionFileFormatter(logging.Formatter):
    _IMPORTANT_PATTERNS = (
        re.compile(r"--- STAGE \d+/\d+:"),
        re.compile(r"REFINEMENT ITERATION"),
        re.compile(r"MINLP COUNTERFACTUAL SEARCH"),
        re.compile(r"MINLP SEARCH DONE"),
    )

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        message = record.getMessage()
        if any(p.search(message) for p in self._IMPORTANT_PATTERNS):
            return "\n\n" + text
        return text


def _configure_logging(verbose: bool) -> None:
    for old_file in LOGS_DIR.glob("execution_logs_*.log"):
        old_file.unlink(missing_ok=True)
    for old_file in LOGS_DIR.glob("workflow_logs_*.log"):
        old_file.unlink(missing_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_log_path = LOGS_DIR / f"execution_logs_{timestamp}.log"
    workflow_log_path = LOGS_DIR / f"workflow_logs_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    execution_handler = logging.FileHandler(execution_log_path, mode="w", encoding="utf-8")
    execution_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    execution_handler.setFormatter(_ExecutionFileFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(execution_handler)

    workflow_logger.handlers.clear()
    workflow_handler = logging.FileHandler(workflow_log_path, mode="w", encoding="utf-8")
    workflow_handler.setLevel(logging.INFO)
    workflow_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    workflow_logger.addHandler(workflow_handler)
    workflow_logger.setLevel(logging.INFO)
    workflow_logger.propagate = False

    logger.info("Execution logs file: %s", execution_log_path)
    logger.info("Workflow logs file: %s", workflow_log_path)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-sample MINLP workbench for continuous regression targets."
        )
    )
    parser.add_argument(
        "--dataset", "-d", choices=list(CONT_DATASETS), default=USER_DATASET,
        help="Continuous-target dataset to load from development/data_cont.",
    )
    parser.add_argument(
        "--sample-index", "-i", type=int, action="append", default=None,
        help="Sample index to inspect (repeatable). Overrides USER_SAMPLE_INDICES.",
    )
    parser.add_argument(
        "--target-y", type=float, default=USER_TARGET_Y_SCALED,
        help="Desired model output on the scaled [0,1] target space.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=USER_EPSILON,
        help="Tolerance around target_y the explainer should respect.",
    )
    parser.add_argument(
        "--target-exemplar-epsilon", type=float,
        default=USER_TARGET_EXEMPLAR_EPSILON,
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
    parser.add_argument(
        "--force-setup", action="store_true",
        help="Re-prepare the dataset and re-train the regression model.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--show-features", action="store_true",
        help="Print the feature index/name table and exit.",
    )
    parser.add_argument(
        "--show-sample", type=int, default=None,
        help="Print the values of a specific test sample and exit.",
    )
    parser.add_argument(
        "--analyze-dataset", dest="analyze_dataset", action="store_true",
        default=True,
        help="Run dataset analysis (plots + summary.txt) before MINLP. Default: on.",
    )
    parser.add_argument(
        "--no-analyze-dataset", dest="analyze_dataset", action="store_false",
        help="Skip the dataset analysis step.",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run only the dataset analysis and exit (no MINLP search).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    ctx = load_cont_context(
        args.dataset,
        force_setup=args.force_setup,
        epochs=args.epochs,
        batch_size=args.batch_size,
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
        sample_indices = [0]

    if args.analyze_dataset or args.analysis_only:
        first_idx = int(sample_indices[0])
        X_test_np = _as_array(ctx.X_test)
        first_sample = (
            X_test_np[first_idx].astype(float)
            if 0 <= first_idx < len(X_test_np) else None
        )
        try:
            exemplar = find_continuous_target_exemplar(
                ctx, target_y=float(args.target_y),
            )
        except Exception as exc:
            logger.warning("Could not pick exemplar for dataset analysis: %s", exc)
            exemplar = None
        run_dataset_analysis(
            ctx,
            sample=first_sample,
            exemplar=exemplar,
        )
        if args.analysis_only:
            return

    results = run_minlp_cont_on_samples(
        ctx,
        sample_indices,
        build_my_priorities,
        target_y=args.target_y,
        epsilon=args.epsilon,
        target_exemplar_epsilon=args.target_exemplar_epsilon,
        shap_approx=args.shap_approx,
        shap_num_samples=args.shap_num_samples,
    )

    n_total = len(results)
    n_within = sum(
        1 for r in results
        if r["metrics"] is not None and r["metrics"]["within_epsilon"]
    )
    rate = (100.0 * n_within / n_total) if n_total else 0.0
    logger.info(
        "Workbench done: %d sample(s); %d within ±%.4f of target (%.1f%%).",
        n_total, n_within, args.epsilon, rate,
    )


if __name__ == "__main__":
    main()
