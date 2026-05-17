"""
Tutorial: MINLSearchExplainer for a continuous target (regression)

What this method is for
-----------------------
MINLSearchExplainer searches for a counterfactual by combining:
  1) a reference dataset (to find a "target exemplar" near your desired target prediction),
  2) Shapley values (to identify important features for moving sample -> exemplar),
  3) constrained optimization (to hit the target within epsilon while respecting priorities).

You will learn how to:
  - define continuous priorities for a regression problem,
  - define categorical priority tables (and how to "forbid" categories if needed),
  - pick a target value and epsilon for a continuous prediction.

Note on scale
-------------
For clarity, this tutorial scales the regression target to [0,1]. That makes:
  - `target` easy to interpret,
  - `epsilon` comparable across datasets.
"""

from __future__ import annotations

import logging
import ssl
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Allow running this example from the repository without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

# Import explainit after logging configuration, because explainit.logging_config sets DEBUG by default.
from explainit.explainers.minlp_search import MINLSearchExplainer
from explainit.priorities import linear, nonlinear


def _stage(title: str) -> None:
    logger.info("")
    logger.info("=" * 88)
    logger.info("STAGE: %s", title)
    logger.info("=" * 88)


def _step(title: str) -> None:
    logger.info("  - STEP: %s", title)


def _as_float01(x: Any) -> float:
    return float(np.asarray(x).squeeze())


def make_exponential_preference(x0: float, x1: float, increasing: bool, a: float = 5.0):
    def pref(x: float) -> float:
        return _as_float01(nonlinear.exponential(x, x0=x0, x1=x1, increasing=increasing, a=a))

    return pref


def make_linear_preference(x0: float, x1: float, increasing: bool):
    def pref(x: float) -> float:
        return _as_float01(linear.basic_linear(x, x0=x0, x1=x1, increasing=increasing))

    return pref


def load_wine_quality_with_one_categorical(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], MinMaxScaler, MinMaxScaler]:
    _step("Download Wine Quality (red) from UCI and add a small categorical feature (AlcoholBand)")

    # Some environments lack CA certificates; use the same unverified SSL approach used elsewhere
    # in this repo's examples (tutorial convenience, not production guidance).
    ssl._create_default_https_context = ssl._create_unverified_context

    primary_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    fallback_urls = [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
    ]

    last_err: Exception | None = None
    df: pd.DataFrame | None = None
    for url in [primary_url, *fallback_urls]:
        try:
            # UCI uses ';' separator; some mirrors are comma-separated.
            try:
                df = pd.read_csv(url, sep=";")
            except Exception:
                df = pd.read_csv(url)
            if "quality" not in df.columns:
                raise ValueError(f"Downloaded dataset missing 'quality' column from {url}")
            break
        except Exception as e:
            last_err = e
            df = None

    if df is None:
        raise RuntimeError(f"Failed to download Wine Quality dataset. Last error: {last_err}")
    if "quality" not in df.columns:
        raise ValueError("Unexpected dataset format: missing 'quality' column")

    y = df["quality"].astype(float).to_numpy()
    X_df = df.drop(columns=["quality"]).copy()

    # IMPORTANT (tutorial convenience):
    # MINLSearchExplainer.find_counterfactuals currently uses a parameter vector `x` whose
    # positions are assumed to match feature indices directly (it indexes x[idx]).
    # To avoid index mismatches, we reorder columns so that the "actionable" features we optimize
    # occupy indices 0..k-1.
    actionable_cols = ["alcohol", "volatile acidity", "sulphates", "residual sugar"]
    remaining_cols = [c for c in X_df.columns if c not in actionable_cols]
    X_df = X_df[actionable_cols + remaining_cols]

    # Derived categorical feature: AlcoholBand by quantiles (0/1/2).
    alcohol_band = pd.qcut(X_df["alcohol"], q=3, labels=[0, 1, 2]).astype(int)

    # Scale continuous features to [0,1], keep AlcoholBand unscaled and append.
    x_scaler = MinMaxScaler()
    X_cont = x_scaler.fit_transform(X_df.to_numpy(dtype=float))
    X = np.hstack([X_cont, alcohol_band.to_numpy(dtype=float).reshape(-1, 1)])

    feature_names = list(X_df.columns) + ["AlcoholBand"]

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=test_size, random_state=random_state
    )
    logger.info("Dataset: %d train / %d test | features=%d", len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test, feature_names, x_scaler, y_scaler


def train_regressor(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    _step("Train a regression model (Ridge regression)")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def choose_sample_and_target(model: Any, X_test: np.ndarray, X_train: np.ndarray) -> tuple[int, np.ndarray, float, float]:
    _step("Pick a sample and choose a realistic continuous target value")
    preds = model.predict(X_test)
    # Avoid extreme samples/targets: MINLP is easier to satisfy when the target isn't too far.
    order = np.argsort(preds)
    sample_idx = int(order[int(0.2 * (len(order) - 1))])
    sample = X_test[sample_idx].copy()
    original_pred = float(preds[sample_idx])

    # Choose a target close to the current prediction (smaller shift => easier optimization).
    desired = float(min(0.95, original_pred + 0.10))

    # Pick the target from the *training* prediction distribution so MINLP can find a target exemplar.
    # Prefer a value that is unique to avoid ties inside MINLSearchExplainer.find_closest_elem().
    train_preds = model.predict(X_train)
    q = desired
    vals, counts = np.unique(train_preds, return_counts=True)
    unique_vals = vals[counts == 1]
    if len(unique_vals) > 0:
        target = float(unique_vals[np.argmin(np.abs(unique_vals - q))])
    else:
        target = q

    if target <= original_pred:
        sample_idx = int(order[int(0.8 * (len(order) - 1))])
        sample = X_test[sample_idx].copy()
        original_pred = float(preds[sample_idx])
        desired = float(max(0.05, original_pred - 0.10))
        q = desired
        if len(unique_vals) > 0:
            target = float(unique_vals[np.argmin(np.abs(unique_vals - q))])
        else:
            target = q

    return sample_idx, sample, original_pred, target


def build_priorities(
    feature_names: list[str],
    sample: np.ndarray,
    X_train: np.ndarray,
    *,
    want_increase: bool,
    numerical_features: list[str],
) -> Dict[str, Any]:
    _step("Define priorities (choose a small actionable subset to keep MINLP fast)")

    fi = {n: i for i, n in enumerate(feature_names)}

    # Only include a handful of numerical indices here.
    # MINLSearchExplainer treats the keys of priorities['numerical'] as the variables to optimize.
    numerical: Dict[int, Dict[str, Any]] = {}

    def add_num(name: str, pref_fn: Any) -> None:
        if name not in fi:
            raise KeyError(f"Unknown feature name: {name}")
        j = fi[name]
        col = X_train[:, j].astype(float)
        numerical[j] = {"min": float(np.min(col)), "max": float(np.max(col)), "function": pref_fn}

    # Domain-motivated monotonic preferences (scaled to [0,1]).
    # We attach preferences only for the feature names passed in `numerical_features`.
    for name in numerical_features:
        if name == "alcohol":
            add_num(name, make_exponential_preference(0.5, 0.9, increasing=want_increase, a=6.0))
        elif name == "volatile acidity":
            add_num(name, make_exponential_preference(0.2, 0.6, increasing=not want_increase, a=6.0))
        elif name == "sulphates":
            add_num(name, make_linear_preference(0.3, 0.8, increasing=want_increase))
        elif name == "citric acid":
            add_num(name, make_linear_preference(0.2, 0.7, increasing=want_increase))
        elif name == "residual sugar":
            add_num(name, make_exponential_preference(0.2, 0.7, increasing=not want_increase, a=5.0))
        else:
            # Fallback: prefer higher values when increasing target, lower values otherwise.
            add_num(name, make_linear_preference(0.2, 0.8, increasing=want_increase))

    # Categorical priorities: AlcoholBand (0/1/2) with weights.
    band_idx = fi["AlcoholBand"]
    categorical: Dict[tuple[int, ...], Dict[tuple[float, ...], float | None]] = {
        (band_idx,): {(0.0,): 0.2, (1.0,): 0.7, (2.0,): 1.0}
    }

    # Optional: to forbid a category for MINLP, set its weight to None (it will be filtered out
    # from the reference dataset before searching). This example keeps everything allowed.
    # If you do forbid, make sure the sample's current category remains allowed.

    return {"numerical": numerical, "categorical": categorical}


def preview_priorities(priorities: Dict[str, Any], feature_names: list[str], sample: np.ndarray) -> None:
    _step("Preview priorities at the current sample")
    for idx, cfg in priorities["numerical"].items():
        name = feature_names[idx]
        w = _as_float01(cfg["function"](float(sample[idx])))
        logger.info("NUM  %-16s idx=%2d | sample=%.3f | bounds=[%.3f, %.3f] | pref=%.3f", name, idx, float(sample[idx]), float(cfg["min"]), float(cfg["max"]), w)

    for group, mapping in priorities["categorical"].items():
        group_names = [feature_names[i] for i in group]
        combo = tuple(float(sample[i]) for i in group)
        w = mapping.get(combo, None)
        logger.info("CAT  %s idx=%s | sample_combo=%s | weight=%s", group_names, list(group), combo, "None" if w is None else f"{float(w):.3f}")


def score_priorities(priorities: Dict[str, Any], x: np.ndarray) -> float:
    """Lightweight score: sum of numerical preference values + categorical weight."""
    total = 0.0
    for idx, cfg in priorities["numerical"].items():
        f = cfg.get("function")
        if f is not None:
            total += float(f(float(x[idx])))
    for group, mapping in priorities["categorical"].items():
        combo = tuple(float(x[i]) for i in group)
        w = mapping.get(combo, 0.0)
        if w is not None:
            total += float(w)
    return float(total)


def _describe_changes(feature_names: list[str], before: np.ndarray, after: np.ndarray) -> list[str]:
    out: list[str] = []
    for i, n in enumerate(feature_names):
        if abs(float(after[i]) - float(before[i])) > 1e-9:
            out.append(f"{n}: {float(before[i]):.3f} -> {float(after[i]):.3f}")
    return out


def main() -> None:
    _stage("1) Data + model (download + train)")
    X_train, X_test, y_train, y_test, feature_names, _, _ = load_wine_quality_with_one_categorical()
    model = train_regressor(X_train, y_train)

    _stage("2) Choose sample + continuous target")
    sample_idx, sample, original_pred, target = choose_sample_and_target(model, X_test, X_train)
    want_increase = target > original_pred
    logger.info("Sample index: %d", sample_idx)
    logger.info("Original prediction (scaled y): %.4f", original_pred)
    logger.info("Target prediction   (scaled y): %.4f", target)
    logger.info("Direction: %s", "increase" if want_increase else "decrease")

    _stage("3) Define priorities (continuous + categorical)")
    # Candidate pool: we will keep only those features that actually differ between
    # the sample and the chosen target exemplar, because MINLSearchExplainer currently
    # computes some coefficients using (target_exemplar[i] - sample[i]) in the denominator.
    # Since we reordered columns in the loader, these occupy indices 0..3.
    candidate_features = ["alcohol", "volatile acidity", "sulphates", "residual sugar"]
    prelim_priorities = build_priorities(
        feature_names,
        sample,
        X_train,
        want_increase=want_increase,
        numerical_features=candidate_features,
    )

    def model_pred(X_batch: Any) -> np.ndarray:
        X_np = np.asarray(X_batch, dtype=float)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        return model.predict(X_np)

    _step("Pre-check: find a target exemplar and keep only numerical features that actually change")
    pre = MINLSearchExplainer(
        model_pred=model_pred,
        priorities=prelim_priorities,
        sample=sample,
        target=target,
        dataset=X_train.astype(float),
        target_exemplar_epsilon=0.3,
        epsilon=0.12,
    )
    try:
        pre.find_closest_elem()
        exemplar = np.asarray(pre.sample_state.target_exemplar, dtype=float)
        fi = {n: i for i, n in enumerate(feature_names)}
        changing = [n for n in candidate_features if abs(float(exemplar[fi[n]]) - float(sample[fi[n]])) > 1e-9]
        logger.info("Numerical features that differ vs. exemplar: %s", changing)
    except Exception as e:
        logger.warning("Could not pre-compute target exemplar (%s).", e)

    # IMPORTANT (workaround for current MINLSearchExplainer implementation):
    # find_counterfactuals uses x[idx] where idx is a *feature index*, so numerical indices
    # must be a dense block 0..k-1. Since our loader reordered columns accordingly, we keep
    # the full contiguous block here.
    varying = candidate_features
    logger.info("Using numerical features in MINLP: %s", varying)

    priorities = build_priorities(
        feature_names,
        sample,
        X_train,
        want_increase=want_increase,
        numerical_features=varying,
    )
    preview_priorities(priorities, feature_names, sample)
    logger.info("Priority score at sample: %.4f", score_priorities(priorities, sample))

    _stage("4) Create MINLP explainer (needs reference dataset)")
    explainer = MINLSearchExplainer(
        model_pred=model_pred,
        priorities=priorities,
        sample=sample,
        target=target,
        dataset=X_train.astype(float),
        target_exemplar_epsilon=0.2,
        epsilon=0.10,
    )

    _stage("5) Run MINLP search for a continuous target")
    _step("Use approximate Shapley values to keep runtime reasonable")
    try:
        cf = explainer.find_counterfactuals(shap_approx=True, num_samples=120)
    except Exception as e:
        logger.warning("MINLP search failed with epsilon=%.3f (%s). Retrying with larger epsilon...", explainer.epsilon, e)
        explainer.epsilon = 0.20
        try:
            cf = explainer.find_counterfactuals(shap_approx=True, num_samples=120)
        except Exception as e2:
            logger.error("MINLP search failed again: %s", e2)
            return

    cf = np.asarray(cf, dtype=float)
    cf_pred = float(model_pred(cf)[0])
    logger.info("")
    logger.info("--- Counterfactual ---")
    logger.info("pred=%.4f | target=%.4f ± %.3f | priority score=%.4f", cf_pred, target, explainer.epsilon, score_priorities(priorities, cf))
    for line in _describe_changes(feature_names, sample, cf):
        logger.info("  %s", line)

    _stage("Done")
    logger.info("Tip: add/remove numerical indices in priorities to trade off quality vs. speed.")


if __name__ == "__main__":
    main()
