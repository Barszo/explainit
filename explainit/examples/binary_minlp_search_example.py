"""
Tutorial: MINLSearchExplainer (MINLP + Shapley-guided counterfactual search)

What this method is for
-----------------------
MINLSearchExplainer aims to produce *few, high-quality* counterfactuals by:
  1) finding a "target exemplar" in a reference dataset that already achieves the desired outcome,
  2) using (approx.) Shapley values to identify which features matter for moving from the sample to that exemplar,
  3) solving an optimization problem over actionable features to hit the target while maximizing your preferences.

When to choose MINLP over Random Search
---------------------------------------
- You can provide a reference dataset in the same feature space as your model input.
- You want a more guided search than pure sampling (often fewer iterations, more structure).
- You can afford extra compute vs. RandomSearchExplainer.

Key concept: priorities (same idea, slightly different conventions)
-------------------------------------------------------------------
Priorities have the same shape: {'numerical': ..., 'categorical': ...}, but MINLP expects a
more explicit "new format" for numerical features:

Numerical / continuous:
  index -> {'min': ..., 'max': ..., 'function': f}
  - f(x) returns a weight in [0,1] (higher = more preferred).
  - If a feature is non-actionable: set function=None and min==max==sample_value.

Categorical:
  group indices -> {(category tuple): weight}
  - For MINLP, you can *remove* a category by leaving it out of the dict, or (in some parts
    of the implementation) mark it as None to filter it out of the dataset pre-search.
  - In this tutorial we use numeric weights, and also show one safe example of forbidding a
    category with None (while keeping the current sample's category allowed).

This file is staged: follow STAGE / STEP logs.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

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


def load_heart_disease_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _step("Load a dataset with mixed feature types (continuous + categorical)")

    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        data = pd.read_csv(url, names=column_names, na_values='?')
    except Exception as e:
        logger.warning("Primary download failed, trying fallback raw GitHub dataset: %s", e)
        fallback_url = "https://raw.githubusercontent.com/rashida048/Datasets/master/heart.csv"
        data = pd.read_csv(fallback_url)
        if 'target' not in data.columns:
            raise ValueError("Dataset does not contain 'target' column")
    data = data.dropna()
    data['target'] = (data['target'] > 0).astype(int)

    X = data.drop(columns=['target'])
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Dataset ready: %d train / %d test, %d features", len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test


def train_binary_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    _step("Train a probabilistic model; MINLP only needs model_pred(X)->probabilities")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
    model.fit(X_train, y_train)
    return model


def make_exponential_preference(x0: float, x1: float, increasing: bool, a: float = 5.0):
    def pref(x: float) -> float:
        return _as_float01(nonlinear.exponential(x, x0=x0, x1=x1, increasing=increasing, a=a))

    return pref


def make_linear_preference(x0: float, x1: float, increasing: bool):
    def pref(x: float) -> float:
        return _as_float01(linear.basic_linear(x, x0=x0, x1=x1, increasing=increasing))

    return pref


def build_priorities(
    feature_names: list[str],
    sample: np.ndarray,
    X_train: pd.DataFrame,
    *,
    target_class: int,
) -> Dict[str, Any]:
    _step("Define priorities (numerical functions + categorical weight tables)")
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}
    prefer_higher_risk = int(target_class) == 1

    # -------------------------------------------------------------------------
    # STEP 3A — Numerical features in "new format" (dict with min/max/function)
    # -------------------------------------------------------------------------
    numerical: Dict[int, Dict[str, Any]] = {}

    def col_minmax(name: str) -> tuple[float, float]:
        col = X_train[name].astype(float)
        return float(col.min()), float(col.max())

    def set_continuous(name: str, *, min_val: float, max_val: float, pref_fn: Any) -> None:
        numerical[feature_indices[name]] = {"min": float(min_val), "max": float(max_val), "function": pref_fn}

    # Align preference direction with the goal:
    # - target_class=0: prefer "healthier" directions (lower age/chol/trestbps/oldpeak, higher thalach)
    # - target_class=1: prefer "riskier" directions (higher age/chol/trestbps/oldpeak, lower thalach)

    # Age
    age_min, age_max = col_minmax("age")
    set_continuous(
        "age",
        min_val=age_min,
        max_val=age_max,
        pref_fn=make_exponential_preference(45.0, 65.0, increasing=prefer_higher_risk, a=6.0),
    )

    # Cholesterol
    chol_min, chol_max = col_minmax("chol")
    set_continuous(
        "chol",
        min_val=chol_min,
        max_val=chol_max,
        pref_fn=make_linear_preference(220.0, 280.0, increasing=prefer_higher_risk),
    )

    # Max heart rate
    thalach_min, thalach_max = col_minmax("thalach")
    set_continuous(
        "thalach",
        min_val=thalach_min,
        max_val=thalach_max,
        pref_fn=make_exponential_preference(120.0, 160.0, increasing=not prefer_higher_risk, a=5.0),
    )

    # Resting blood pressure
    trestbps_min, trestbps_max = col_minmax("trestbps")
    set_continuous(
        "trestbps",
        min_val=trestbps_min,
        max_val=trestbps_max,
        pref_fn=make_exponential_preference(120.0, 150.0, increasing=prefer_higher_risk, a=4.0),
    )

    # ST depression induced by exercise relative to rest (oldpeak)
    oldpeak_min, oldpeak_max = col_minmax("oldpeak")
    set_continuous(
        "oldpeak",
        min_val=oldpeak_min,
        max_val=oldpeak_max,
        pref_fn=make_exponential_preference(0.5, 2.0, increasing=prefer_higher_risk, a=5.0),
    )

    # -------------------------------------------------------------------------
    # STEP 3B — Categorical features: weight tables
    #
    # Keep weights numeric for stable optimization. If you want to forbid a category for MINLP,
    # use `None` (the implementation treats None as "remove those rows from the reference dataset"),
    # but make sure the sample's own category remains allowed (numeric), otherwise the search
    # may become infeasible.
    # -------------------------------------------------------------------------
    categorical: Dict[tuple[int, ...], Dict[tuple[float, ...], float]] = {}

    cp_idx = feature_indices["cp"]
    cp_values = sorted({float(v) for v in X_train["cp"].unique()})
    categorical[(cp_idx,)] = {(v,): (1.0 if int(v) == 3 else 0.7 if int(v) == 4 else 0.4) for v in cp_values}

    thal_idx = feature_indices["thal"]
    thal_values = sorted({float(v) for v in X_train["thal"].unique()})
    categorical[(thal_idx,)] = {(v,): (1.0 if int(v) == 3 else 0.6) for v in thal_values}

    # Grouped categorical preference: (exang, slope)
    exang_idx = feature_indices["exang"]
    slope_idx = feature_indices["slope"]
    combos = sorted({(float(a), float(b)) for a, b in zip(X_train["exang"], X_train["slope"])})
    categorical[(exang_idx, slope_idx)] = {
        (exang, slope): (
            1.0 if (int(exang) == 0 and int(slope) == 2) else 0.8 if (int(exang) == 0 and int(slope) == 3) else 0.4
        )
        for exang, slope in combos
    }

    return {"numerical": numerical, "categorical": categorical}


def preview_priorities(priorities: Dict[str, Any], feature_names: list[str], sample: np.ndarray) -> None:
    _step("Preview priorities at the current sample (sanity-check indices + weights)")

    logger.info("Index mapping (first 13): %s", {i: n for i, n in enumerate(feature_names[:13])})

    for idx, cfg in priorities["numerical"].items():
        name = feature_names[idx] if idx < len(feature_names) else str(idx)
        f = cfg.get("function")
        if f is None:
            logger.info(
                "NUM  %-10s idx=%2d | non-actionable (function=None), fixed bounds=[%.3f, %.3f]",
                name,
                idx,
                float(cfg["min"]),
                float(cfg["max"]),
            )
        else:
            w = _as_float01(f(float(sample[idx])))
            logger.info(
                "NUM  %-10s idx=%2d | sample=%.3f | bounds=[%.3f, %.3f] | preference=%.3f",
                name,
                idx,
                float(sample[idx]),
                float(cfg["min"]),
                float(cfg["max"]),
                float(w),
            )

    for group, mapping in priorities["categorical"].items():
        group_names = [feature_names[i] for i in group]
        sample_combo = tuple(float(sample[i]) for i in group)
        w = mapping.get(sample_combo, None)
        none_count = sum(1 for v in mapping.values() if v is None)
        logger.info(
            "CAT  %s idx=%s | sample_combo=%s | weight=%s | forbidden(None)=%d/%d",
            group_names,
            list(group),
            sample_combo,
            "None" if w is None else f"{float(w):.3f}",
            none_count,
            len(mapping),
        )


def _describe_changes(feature_names: list[str], before: np.ndarray, after: np.ndarray) -> list[str]:
    changes: list[str] = []
    for i, name in enumerate(feature_names):
        b = float(before[i])
        a = float(after[i])
        if abs(a - b) > 1e-9:
            changes.append(f"{name}: {b:.3f} -> {a:.3f}")
    return changes


def main() -> None:
    """Main example demonstrating MINLP-based binary classification counterfactuals."""

    _stage("1) Data + model")
    X_train, X_test, y_train, y_test = load_heart_disease_data()
    model = train_binary_classifier(X_train, y_train)

    feature_names = list(X_train.columns)

    _stage("2) Choose a sample to explain + define the goal")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_classes = (test_probs > 0.5).astype(int)

    # For a stable tutorial run, prefer starting from a class-1 sample and flip to class 0
    # (this aligns with "healthier" priorities and usually yields counterfactuals faster).
    pos_indices = np.where(test_classes == 1)[0]
    if len(pos_indices) > 0:
        sample_idx = int(pos_indices[0])
        target_class = 0
    else:
        sample_idx = 0
        target_class = 1

    sample = X_test.iloc[sample_idx].to_numpy(dtype=float)
    original_prob = float(test_probs[sample_idx])
    original_class = int(original_prob > 0.5)

    logger.info("Sample index: %d", sample_idx)
    logger.info("Original p(class=1): %.4f (class=%d @ threshold=0.5)", float(original_prob), original_class)
    logger.info("Goal: find counterfactuals for target_class=%d", target_class)

    _stage("3) Define priorities (continuous + categorical)")
    priorities = build_priorities(feature_names, sample, X_train, target_class=target_class)
    preview_priorities(priorities, feature_names, sample)
    logger.info(
        "Priorities defined: %d numerical entries, %d categorical groups",
        len(priorities["numerical"]),
        len(priorities["categorical"]),
    )

    _stage("4) Create the MINLP explainer (requires a reference dataset)")
    def model_pred(X_batch: np.ndarray) -> np.ndarray:
        """Return probability of positive class."""
        return model.predict_proba(X_batch)[:, 1]

    try:
        explainer = MINLSearchExplainer(
            model_pred=model_pred,
            priorities=priorities,
            sample=sample,
            target=0.7,  # target probability used to find a target exemplar (binary helper will override)
            dataset=X_train.to_numpy(dtype=float),
            target_exemplar_epsilon=0.05,
            epsilon=0.05
        )
    except Exception as e:
        logger.error("Failed to create explainer: %s", e)
        return

    _stage("5) (Optional) See how priorities filter the reference dataset")
    try:
        filtered = explainer.get_rows_in_priorities()
        logger.info("Filtered dataset size: %d -> %d rows", X_train.shape[0], filtered.shape[0])
    except Exception as e:
        logger.warning("Could not filter dataset (priorities may be too restrictive): %s", e)

    _stage("6) Run MINLP search for binary counterfactuals")
    _step("Compute Shapley values (exact or approximate) and solve an optimization problem")
    try:
        cfs, predictions, scores, iterations = explainer.find_counterfactuals_for_binary(
            target_class=target_class,
            threshold=0.5,
            expected_counterfactuals=3,
            max_iterations=400,
            shap_approx=False,
            return_top_n=3,
        )
    except Exception as e:
        logger.error("Error during counterfactual search: %s", e)
        return

    if len(cfs) == 0:
        logger.warning("No valid counterfactuals found. Try loosening bounds or increasing max_iterations.")
        return

    for k, (cf, pred, score, iteration) in enumerate(zip(cfs, predictions, scores, iterations), start=1):
        logger.info("")
        logger.info("--- Counterfactual %d (iteration %d) ---", k, int(iteration))
        logger.info("p(class=1): %.4f | preference score: %.4f", float(pred), float(score))
        for line in _describe_changes(feature_names, sample, np.asarray(cf, dtype=float)):
            logger.info("  %s", line)

    _stage("Done")
    logger.info("Tip: if results look odd, visualize/print your priorities first and confirm feature indices.")


if __name__ == '__main__':
    main()
