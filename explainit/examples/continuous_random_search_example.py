"""
Tutorial: RandomSearchExplainer for a continuous target (regression)

What this method is for
-----------------------
RandomSearchExplainer searches for counterfactuals by *sampling* candidate feature vectors
subject to your constraints, then keeping those whose model prediction is close to a chosen
continuous target value.

You will learn how to:
  - set priorities for continuous features (min/max bounds + preference functions),
  - set priority values for categorical features (explicit weight tables),
  - pick a realistic regression target and tolerance (epsilon),
  - run the random search and interpret the results.

Important implementation detail (RandomSearchExplainer)
-------------------------------------------------------
The current implementation starts each candidate from an all-zeros vector and only fills
indices present in `priorities['numerical']` and `priorities['categorical']`.
Therefore, for regression use-cases you should cover *every feature index* in
`priorities['numerical']`:
  - actionable feature: {idx: {'min': ..., 'max': ..., 'function': f}}
  - non-actionable feature: {idx: 0}  (it will be fixed to the original sample value)
Categorical groups then overwrite their indices with sampled category combinations.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
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
from explainit.explainers.random_search import RandomSearchExplainer
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
    """f(x) -> [0,1] using a smooth exponential transition."""

    def pref(x: float) -> float:
        return _as_float01(nonlinear.exponential(x, x0=x0, x1=x1, increasing=increasing, a=a))

    return pref


def make_linear_preference(x0: float, x1: float, increasing: bool):
    """f(x) -> [0,1] using a simple linear transition."""

    def pref(x: float) -> float:
        return _as_float01(linear.basic_linear(x, x0=x0, x1=x1, increasing=increasing))

    return pref


def load_california_housing_with_one_categorical(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], MinMaxScaler, MinMaxScaler]:
    _step("Download California Housing and add a small categorical feature (MedIncBand)")

    data = fetch_california_housing(as_frame=True)
    X_df: pd.DataFrame = data.data.copy()
    y: pd.Series = data.target.copy()

    # Derive a simple categorical feature from a continuous one:
    # 0=low income, 1=mid income, 2=high income (by quantiles).
    medinc_band = pd.qcut(X_df["MedInc"], q=3, labels=[0, 1, 2]).astype(int)

    # Scale continuous features to [0,1] to make min/max and priority functions easy to read.
    x_scaler = MinMaxScaler()
    X_cont = x_scaler.fit_transform(X_df.values.astype(float))

    # Keep the categorical band unscaled and append as the last column.
    X = np.hstack([X_cont, medinc_band.to_numpy(dtype=float).reshape(-1, 1)])
    feature_names = list(X_df.columns) + ["MedIncBand"]

    # Scale the regression target to [0,1] as well (so "epsilon" is interpretable).
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.to_numpy(dtype=float).reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=test_size, random_state=random_state
    )
    logger.info("Dataset: %d train / %d test | features=%d", len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test, feature_names, x_scaler, y_scaler


def train_regressor(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    _step("Train a regression model (GradientBoostingRegressor) on the downloaded data")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def choose_sample_and_target(
    model: Any, X_test: np.ndarray, *, target_quantile: float = 0.8
) -> tuple[int, np.ndarray, float, float]:
    _step("Pick a sample and choose a realistic continuous target value")

    preds = model.predict(X_test)
    # Choose a "low" predicted sample, then target a "higher" value (quantile).
    sample_idx = int(np.argmin(preds))
    sample = X_test[sample_idx].copy()
    original_pred = float(preds[sample_idx])
    target = float(np.quantile(preds, target_quantile))

    # If the chosen sample is already high, flip the direction (target a lower value).
    if target <= original_pred:
        sample_idx = int(np.argmax(preds))
        sample = X_test[sample_idx].copy()
        original_pred = float(preds[sample_idx])
        target = float(np.quantile(preds, 1.0 - target_quantile))

    return sample_idx, sample, original_pred, target


def build_priorities(
    feature_names: list[str],
    sample: np.ndarray,
    X_train: np.ndarray,
    *,
    want_increase: bool,
) -> Dict[str, Any]:
    _step("Define priorities: continuous functions + a categorical weight table")

    idx = {n: i for i, n in enumerate(feature_names)}
    n_features = len(feature_names)

    # Cover all indices so unspecified features aren't forced to 0.
    numerical: Dict[int, Any] = {i: 0 for i in range(n_features)}

    def add_num(name: str, pref_fn: Any) -> None:
        j = idx[name]
        col = X_train[:, j].astype(float)
        numerical[j] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "function": pref_fn,
        }

    # With MinMax scaling, most continuous feature ranges are ~[0,1].
    # We'll pick a few actionable features and keep the rest fixed to the sample.
    add_num("MedInc", make_exponential_preference(0.4, 0.8, increasing=want_increase, a=6.0))
    add_num("AveRooms", make_linear_preference(0.3, 0.8, increasing=want_increase))
    add_num("HouseAge", make_linear_preference(0.2, 0.8, increasing=want_increase))
    add_num("AveOccup", make_exponential_preference(0.2, 0.7, increasing=not want_increase, a=5.0))

    # Make one feature explicitly non-actionable (kept constant).
    numerical[idx["Latitude"]] = 0

    # Categorical priorities: MedIncBand (0,1,2).
    # NOTE: this is a *derived* categorical feature used only to demonstrate how to specify weights.
    band_idx = idx["MedIncBand"]
    categorical: Dict[tuple[int, ...], Dict[tuple[Any, ...], float]] = {
        (band_idx,): {
            (0.0,): 0.2,
            (1.0,): 0.7,
            (2.0,): 1.0,
        }
    }

    # If we're trying to increase the prediction, forbid the lowest income band.
    if want_increase:
        categorical[(band_idx,)][(0.0,)] = 0.0

    return {"numerical": numerical, "categorical": categorical}


def preview_priorities(priorities: Dict[str, Any], feature_names: list[str], sample: np.ndarray) -> None:
    _step("Preview priorities at the current sample (sanity-check indices + weights)")
    logger.info("Index mapping (first 12): %s", {i: n for i, n in enumerate(feature_names[:12])})

    fixed = 0
    actionable = []
    for i, cfg in priorities["numerical"].items():
        name = feature_names[i]
        if isinstance(cfg, dict):
            actionable.append(i)
            w = _as_float01(cfg["function"](float(sample[i])))
            logger.info("NUM  %-12s idx=%2d | sample=%.3f | bounds=[%.3f, %.3f] | pref=%.3f", name, i, float(sample[i]), float(cfg["min"]), float(cfg["max"]), w)
        else:
            fixed += 1
    logger.info("Numerical non-actionable features fixed to sample value: %d/%d", fixed, len(priorities["numerical"]))
    logger.info("Actionable continuous features: %s", sorted(actionable))

    for group, mapping in priorities["categorical"].items():
        group_names = [feature_names[j] for j in group]
        combo = tuple(float(sample[j]) for j in group)
        w = float(mapping.get(combo, 0.0))
        allowed = sum(1 for v in mapping.values() if v != 0.0)
        logger.info("CAT  %s idx=%s | sample_combo=%s | weight=%.3f | allowed=%d/%d (0=forbidden)", group_names, list(group), combo, w, allowed, len(mapping))


def _describe_changes(feature_names: list[str], before: np.ndarray, after: np.ndarray) -> list[str]:
    out: list[str] = []
    for i, n in enumerate(feature_names):
        if abs(float(after[i]) - float(before[i])) > 1e-9:
            out.append(f"{n}: {float(before[i]):.3f} -> {float(after[i]):.3f}")
    return out


def main() -> None:
    _stage("1) Data + model (download + train)")
    X_train, X_test, y_train, y_test, feature_names, _, _ = load_california_housing_with_one_categorical()
    model = train_regressor(X_train, y_train)

    _stage("2) Choose sample + continuous target")
    sample_idx, sample, original_pred, target = choose_sample_and_target(model, X_test, target_quantile=0.8)
    want_increase = target > original_pred
    logger.info("Sample index: %d", sample_idx)
    logger.info("Original prediction (scaled y): %.4f", original_pred)
    logger.info("Target prediction   (scaled y): %.4f", target)
    logger.info("Direction: %s", "increase" if want_increase else "decrease")

    _stage("3) Define priorities (continuous + categorical)")
    priorities = build_priorities(feature_names, sample, X_train, want_increase=want_increase)
    preview_priorities(priorities, feature_names, sample)

    def model_pred(X_batch: np.ndarray) -> np.ndarray:
        return model.predict(X_batch)

    _stage("4) Create explainer + (optional) visualize priorities")
    explainer = RandomSearchExplainer(model_pred=model_pred, priorities=priorities, sample=sample, target=target)
    try:
        viz_priorities = {
            "numerical": {i: cfg for i, cfg in priorities["numerical"].items() if isinstance(cfg, dict)},
            "categorical": priorities["categorical"],
        }
        viz_explainer = RandomSearchExplainer(model_pred=model_pred, priorities=viz_priorities, sample=sample, target=target)
        viz_explainer.display_priorities(exemplar=sample)
    except Exception as e:
        logger.warning("Could not display priorities: %s", e)

    _stage("5) Run random search for continuous target")
    _step("Keep candidates whose prediction is within epsilon of the target")
    epsilon = 0.03
    # RandomSearchExplainer.generate_random_samples reads this attribute (implementation detail).
    explainer.n_candidates_per_cf = 3
    cfs, preds, scores, iters = explainer.generate_random_samples(
        expected_counterfactuals=3,
        max_iterations=15000,
        epsilon=epsilon,
        random_seed=42,
        use_monte_carlo=True,
        max_tries=200,
    )

    if len(cfs) == 0:
        logger.warning("No counterfactuals found. Try increasing max_iterations or epsilon=%.3f.", epsilon)
        return

    for k, (cf, pred, score, iteration) in enumerate(zip(cfs, preds, scores, iters), start=1):
        logger.info("")
        logger.info("--- Counterfactual %d ---", k)
        logger.info("pred=%.4f | target=%.4f ± %.3f | preference score=%.4f | iterations=%d", float(pred), target, epsilon, float(score), int(iteration))
        for line in _describe_changes(feature_names, sample, np.asarray(cf, dtype=float)):
            logger.info("  %s", line)
        logger.info("Preference breakdown: %s", explainer.get_preference_breakdown(cf))

    _stage("Done")
    logger.info("Tip: tighten min/max bounds for more realistic changes; loosen epsilon to get results faster.")


if __name__ == "__main__":
    main()
