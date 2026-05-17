"""
Tutorial: RandomSearchExplainer (preference-guided random sampling)

What this method is for
-----------------------
RandomSearchExplainer is a flexible baseline for generating counterfactuals by *sampling* candidate
feature vectors and keeping those that:
  1) satisfy your *actionability* constraints (bounds + categorical allowances), and
  2) achieve a target prediction (e.g. flip a binary classifier), and
  3) score well under your *priorities* (soft preferences).

Key concept: priorities
-----------------------
You provide a `priorities` dictionary with two parts:

1) Continuous / numerical features (index -> config)
   - Use a dict: {'min': <hard lower bound>, 'max': <hard upper bound>, 'function': f}
   - f(x) must return a weight in [0, 1]; higher means "more preferred".
   - If a feature is *non-actionable*: set its value to 0 (or None) and it will be fixed to the
     original sample value.

2) Categorical features (group of indices -> (category tuple -> weight))
   - The group key is a tuple of indices, e.g. (cp_idx,) or (exang_idx, slope_idx).
   - Each possible category combination maps to a weight in [0, 1].
   - Weight 0 means "forbidden" (RandomSearchExplainer filters those out).

This file is intentionally "staged": follow the STAGE / STEP logs from top to bottom.
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
    """Convert helper outputs (often np.ndarray) into a scalar float."""
    return float(np.asarray(x).squeeze())


def load_heart_disease_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _step("Load a dataset with mixed feature types (continuous + categorical)")

    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    try:
        data = pd.read_csv(url, names=column_names, na_values="?")
    except Exception as e:
        logger.warning("Primary download failed, using fallback dataset: %s", e)
        fallback_url = "https://raw.githubusercontent.com/rashida048/Datasets/master/heart.csv"
        data = pd.read_csv(fallback_url)
        if "target" not in data.columns:
            target_col = [c for c in data.columns if "target" in c.lower() or "disease" in c.lower()][0]
            data = data.rename(columns={target_col: "target"})

    data = data.dropna()
    data["target"] = (data["target"] > 0).astype(int)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Dataset ready: %d train / %d test, %d features", len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test


def train_binary_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    _step("Train any probabilistic model (we only need predict_proba for class 1)")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
    model.fit(X_train, y_train)
    return model


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


def build_priorities(
    feature_names: list[str],
    sample: np.ndarray,
    X_train: pd.DataFrame,
    *,
    target_class: int,
) -> Dict[str, Any]:
    _step("Define priorities (hard bounds + soft preferences) for numerical and categorical features")

    # Important: priorities are indexed by *feature position* in the numpy array.
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}
    n_features = len(feature_names)

    # -------------------------------------------------------------------------
    # STEP 3A — Continuous features: hard constraints + preference functions
    #
    # - 'min'/'max' are hard constraints (the explainer never samples outside)
    # - 'function' is a soft preference in [0,1] (higher = more preferred)
    # -------------------------------------------------------------------------
    # IMPORTANT (RandomSearchExplainer implementation detail):
    # The underlying generator starts from an all-zeros vector and only fills indices present in
    # priorities['numerical'] and priorities['categorical'].
    # Therefore, you should cover *every* feature index:
    #   - either define it as actionable (dict with min/max/function),
    #   - or mark it as non-actionable (0/None) so it gets fixed to the sample value.
    numerical: Dict[int, Any] = {i: 0 for i in range(n_features)}

    def add_continuous(
        name: str,
        *,
        min_val: float,
        max_val: float,
        preference: Any,
    ) -> None:
        idx = feature_indices[name]
        numerical[idx] = {"min": float(min_val), "max": float(max_val), "function": preference}

    # Use training-set ranges as safe bounds (a common starting point).
    def col_minmax(name: str) -> tuple[float, float]:
        col = X_train[name].astype(float)
        return float(col.min()), float(col.max())

    # Choose preference direction to support the goal:
    # - if target_class=0 (want "healthier"): prefer lower age/chol/trestbps and higher thalach
    # - if target_class=1 (want "riskier"): prefer higher age/chol/trestbps and lower thalach
    prefer_higher_risk = int(target_class) == 1

    # Example 1: age
    age_min, age_max = col_minmax("age")
    add_continuous(
        "age",
        min_val=age_min,
        max_val=age_max,
        preference=make_exponential_preference(
            x0=45.0,
            x1=65.0,
            increasing=prefer_higher_risk,  # riskier => prefer older
            a=6.0,
        ),
    )

    # Example 2: cholesterol
    chol_min, chol_max = col_minmax("chol")
    add_continuous(
        "chol",
        min_val=chol_min,
        max_val=chol_max,
        preference=make_linear_preference(
            x0=220.0,
            x1=280.0,
            increasing=prefer_higher_risk,  # riskier => prefer higher chol
        ),
    )

    # Example 3: max heart rate (thalach)
    thalach_min, thalach_max = col_minmax("thalach")
    add_continuous(
        "thalach",
        min_val=thalach_min,
        max_val=thalach_max,
        preference=make_exponential_preference(
            x0=120.0,
            x1=160.0,
            increasing=not prefer_higher_risk,  # healthier => prefer higher thalach
            a=5.0,
        ),
    )

    # Optional extra continuous feature to make flips easier: resting blood pressure (trestbps)
    trestbps_min, trestbps_max = col_minmax("trestbps")
    add_continuous(
        "trestbps",
        min_val=trestbps_min,
        max_val=trestbps_max,
        preference=make_exponential_preference(
            x0=120.0,
            x1=150.0,
            increasing=prefer_higher_risk,  # riskier => prefer higher trestbps
            a=4.0,
        ),
    )

    # Non-actionable continuous feature: set priority to 0 (fixed to the sample value).
    # RandomSearchExplainer replaces 0/None with the original sample value internally.
    numerical[feature_indices["oldpeak"]] = 0

    # -------------------------------------------------------------------------
    # STEP 3B — Categorical features: explicit weight tables over allowed values
    #
    # - Keys are feature-index tuples (groups)
    # - Values are dicts: (category tuple) -> weight in [0,1]
    # - Weight 0 forbids a category (it will be removed during filtering)
    # -------------------------------------------------------------------------
    categorical: Dict[tuple[int, ...], Dict[tuple[Any, ...], float]] = {}

    # Single categorical variable: cp (chest pain type, encoded as ints in this dataset).
    cp_idx = feature_indices["cp"]
    cp_values = sorted({int(v) for v in X_train["cp"].unique()})
    categorical[(cp_idx,)] = {
        (v,): (1.0 if v == 3 else 0.7 if v == 4 else 0.4) for v in cp_values
    }

    # Single categorical variable with an explicit "forbidden" value example:
    # forbid a category *different from the current sample value*.
    thal_idx = feature_indices["thal"]
    thal_values = sorted({int(v) for v in X_train["thal"].unique()})
    sample_thal = int(sample[thal_idx])
    forbid_thal = next((v for v in thal_values if v != sample_thal), None)
    categorical[(thal_idx,)] = {
        (v,): (0.0 if (forbid_thal is not None and v == forbid_thal) else (1.0 if v == 3 else 0.6))
        for v in thal_values
    }

    # Grouped categorical preference: treat (exang, slope) as a linked decision.
    # This is useful when combinations matter more than individual values.
    exang_idx = feature_indices["exang"]
    slope_idx = feature_indices["slope"]
    combos = sorted({(int(a), int(b)) for a, b in zip(X_train["exang"], X_train["slope"])})
    # Forbid one combination (again, choose one that's not the sample's current combo).
    sample_exang_slope = (int(sample[exang_idx]), int(sample[slope_idx]))
    forbidden_combo = next((c for c in combos if c != sample_exang_slope), None)

    categorical[(exang_idx, slope_idx)] = {
        (exang, slope): (
            1.0 if (exang == 0 and slope == 2) else
            0.8 if (exang == 0 and slope == 3) else
            0.0 if (forbidden_combo is not None and (exang, slope) == forbidden_combo) else
            0.4
        )
        for exang, slope in combos
    }

    return {"numerical": numerical, "categorical": categorical}


def preview_priorities(priorities: Dict[str, Any], feature_names: list[str], sample: np.ndarray) -> None:
    _step("Preview priorities at the current sample (sanity-check indices + weights)")

    logger.info("Index mapping (first 13): %s", {i: n for i, n in enumerate(feature_names[:13])})

    fixed_count = 0
    actionable_numeric: list[int] = []
    for idx, cfg in priorities["numerical"].items():
        name = feature_names[idx] if idx < len(feature_names) else str(idx)
        if isinstance(cfg, dict):
            actionable_numeric.append(idx)
            f = cfg.get("function")
            w = float("nan") if f is None else _as_float01(f(float(sample[idx])))
            logger.info(
                "NUM  %-10s idx=%2d | sample=%.3f | bounds=[%.3f, %.3f] | preference=%.3f",
                name,
                idx,
                float(sample[idx]),
                float(cfg["min"]),
                float(cfg["max"]),
                float(w),
            )
        else:
            fixed_count += 1
    logger.info("Numerical non-actionable features fixed to sample value: %d/%d", fixed_count, len(priorities["numerical"]))
    logger.info("Actionable continuous features: %s", sorted(actionable_numeric))

    for group, mapping in priorities["categorical"].items():
        group_names = [feature_names[i] for i in group]
        sample_combo = tuple(sample[i] for i in group)
        w = float(mapping.get(sample_combo, 0.0))
        allowed = sum(1 for v in mapping.values() if v != 0)
        logger.info(
            "CAT  %s idx=%s | sample_combo=%s | weight=%.3f | allowed=%d/%d (weight 0 = forbidden)",
            group_names,
            list(group),
            sample_combo,
            w,
            allowed,
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
    _stage("1) Data + model")
    X_train, X_test, y_train, y_test = load_heart_disease_data()
    model = train_binary_classifier(X_train, y_train)

    _stage("2) Choose a sample to explain + define the goal")
    feature_names = list(X_train.columns)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_classes = (test_probs > 0.5).astype(int)

    # For a stable tutorial run, prefer starting from a class-1 sample and flip to class 0
    # (this aligns with "healthier" preferences and usually yields counterfactuals faster).
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
    logger.info("Original p(class=1): %.4f (class=%d @ threshold=0.5)", original_prob, original_class)
    logger.info("Goal: find counterfactuals for target_class=%d", target_class)

    _stage("3) Define priorities (continuous + categorical)")
    priorities = build_priorities(feature_names, sample, X_train, target_class=target_class)
    preview_priorities(priorities, feature_names, sample)
    logger.info(
        "Numerical priorities cover all indices (RandomSearchExplainer fills only what you specify); "
        "categorical groups override their indices."
    )
    logger.info("Categorical priority groups: %s", list(priorities["categorical"].keys()))
    logger.info("Note: categorical weight 0 means 'forbidden' in RandomSearchExplainer.")

    def model_pred(X_batch: np.ndarray) -> np.ndarray:
        return model.predict_proba(X_batch)[:, 1]

    _stage("4) Create the explainer (it will filter priorities internally)")
    explainer = RandomSearchExplainer(
        model_pred=model_pred,
        priorities=priorities,
        sample=sample,
        target=1,
    )

    logger.info("Filtered numerical priorities keys: %s", sorted(explainer.filtered_priorities["numerical"].keys()))
    logger.info("Filtered categorical groups: %s", list(explainer.filtered_priorities["categorical"].keys()))

    _stage("5) (Optional) Visualize your priorities to sanity-check them")
    logger.info("If matplotlib is available, plots will be saved under `images/`.")
    try:
        viz_priorities = {
            "numerical": {idx: cfg for idx, cfg in priorities["numerical"].items() if isinstance(cfg, dict)},
            "categorical": priorities["categorical"],
        }
        viz_explainer = RandomSearchExplainer(
            model_pred=model_pred,
            priorities=viz_priorities,
            sample=sample,
            target=1,
        )
        viz_explainer.display_priorities(exemplar=sample)
    except Exception as e:
        logger.warning("Could not display priorities (matplotlib may not be available): %s", e)

    _stage("6) Run Random Search and inspect counterfactuals")
    _step("Generate candidates with Monte Carlo sampling, keep those that hit the target, rank by preference")
    cfs, predictions, scores, iterations = explainer.generate_for_binary(
        expected_counterfactuals=3,
        max_iterations=20000,
        target_class=target_class,
        threshold=0.5,
        random_seed=42,
        use_monte_carlo=True,
        max_tries=200,
        n_candidates_per_cf=3,
    )

    if len(cfs) == 0:
        logger.warning("No counterfactuals were found with the current priorities and search settings.")
        logger.warning("Try: increase max_iterations, relax threshold, or soften bounds / categorical restrictions.")
        return

    for k, (cf, pred, score, iteration) in enumerate(zip(cfs, predictions, scores, iterations), start=1):
        logger.info("")
        logger.info("--- Counterfactual %d ---", k)
        logger.info("p(class=1): %.4f | preference score: %.4f | iterations: %d", float(pred), float(score), int(iteration))
        for line in _describe_changes(feature_names, sample, np.asarray(cf, dtype=float)):
            logger.info("  %s", line)
        logger.info("Preference breakdown (per feature/group): %s", explainer.get_preference_breakdown(cf))

    _stage("Done")
    logger.info("Tip: iterate on priorities first (plots + breakdown), then increase search budget.")


if __name__ == '__main__':
    main()
