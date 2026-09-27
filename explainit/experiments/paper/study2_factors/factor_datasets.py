"""Badania II: materialise dataset variants for the factor studies.

Each factor answers "what does X do to the method?" by rebuilding the stage-1
dataset with one property changed, retraining the stage-2 model on it, and
registering the variant under its own dataset key so the rest of the pipeline
(priority sets, selection, runners, registry) works unchanged.

Factors:

``train_size``    fraction of the training rows kept (learning-set size)
``n_features``    keep only the k most important features (dataset width);
                  importance is a permutation score of the baseline model
``feature_noise`` additive Gaussian noise on the (scaled) numerical features
``label_noise``   additive Gaussian noise on the training target
``target_skew``   resample the training rows so the target distribution is
                  skewed (the regression analogue of class imbalance)
``numerical_only`` drop every categorical group (feature-type factor)

Variants are written as ordinary ``data/<variant_key>/data.pkl`` +
``models/<variant_key>/model.keras`` pairs, so a study config refers to them
by key exactly like a base dataset. Priority sets are inherited from the base
dataset: the variant key is registered in ``PRIORITY_SETS`` pointing at the
same sets, minus features that a variant removed.

Usage::

    python -m explainit.experiments.paper.study2_factors.factor_datasets --list
    python -m explainit.experiments.paper.study2_factors.factor_datasets \
        --datasets auto_mpg carseats --factors train_size feature_noise
    python -m explainit.experiments.paper.study2_factors.factor_datasets \
        --datasets auto_mpg --factors all --force
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.continuous_minlp.data_setup import DATA_DIR  # noqa: E402
from explainit.experiments.continuous_minlp.model_setup import (  # noqa: E402
    MODEL_BUILDERS,
    MODELS_DIR,
    build_default_regressor,
)
from explainit.experiments.continuous_minlp.priority_sets import (  # noqa: E402
    DATASET_SPECS,
)

logger = logging.getLogger("explainit.experiments.paper.study2_factors.factors")

#: Levels of each factor. The first level of every factor is the baseline and
#: is *not* materialised (it is the untouched dataset).
FACTOR_LEVELS: Dict[str, List[Any]] = {
    "train_size": [1.0, 0.5, 0.25, 0.10],
    "n_features": ["all", 0.75, 0.50],
    "feature_noise": [0.0, 0.1, 0.25, 0.5],
    "label_noise": [0.0, 0.05, 0.10, 0.20],
    "target_skew": ["none", "low_heavy", "high_heavy"],
    "numerical_only": [False, True],
}


def variant_key(dataset_key: str, factor: str, level: Any, seed: int = 0) -> str:
    token = str(level).replace(".", "p").replace("-", "m")
    # Seed 0 keeps the original key so existing results stay addressable.
    suffix = "" if int(seed) == 0 else f"_s{int(seed)}"
    return f"{dataset_key}__{factor}_{token}{suffix}"


def load_base(dataset_key: str) -> Dict[str, Any]:
    path = DATA_DIR / dataset_key / "data.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Base dataset '{dataset_key}' not prepared: {path} is missing. "
            f"Run data_setup.py first."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Factor implementations. Each returns a new payload dict (deep-copied).
# ---------------------------------------------------------------------------


def _subsample_train(payload: Dict[str, Any], fraction: float, rng) -> Dict[str, Any]:
    out = copy.deepcopy(payload)
    n = len(out["X_train"])
    keep = max(10, int(round(float(fraction) * n)))
    idx = rng.choice(n, size=keep, replace=False)
    _take_train_rows(out, idx)
    return out


def _take_train_rows(payload: Dict[str, Any], idx) -> None:
    """Index every row-aligned training view in place."""
    payload["X_train"] = np.asarray(payload["X_train"])[idx]
    payload["y_train"] = np.asarray(payload["y_train"])[idx]
    if "X_train_raw" in payload and payload["X_train_raw"] is not None:
        raw = payload["X_train_raw"]
        payload["X_train_raw"] = raw.iloc[idx] if hasattr(raw, "iloc") else raw[idx]
    if "y_train_raw" in payload and payload["y_train_raw"] is not None:
        payload["y_train_raw"] = np.asarray(payload["y_train_raw"])[idx]


def _numerical_column_indices(payload: Dict[str, Any]) -> List[int]:
    names = list(payload["feature_names"])
    numerical = list(payload.get("numerical_features") or [])
    return [names.index(n) for n in numerical if n in names]


def _add_feature_noise(payload: Dict[str, Any], sigma: float, rng) -> Dict[str, Any]:
    out = copy.deepcopy(payload)
    cols = _numerical_column_indices(out)
    for split in ("X_train", "X_test"):
        arr = np.asarray(out[split], dtype=float).copy()
        arr[:, cols] += rng.normal(0.0, float(sigma), size=arr[:, cols].shape)
        out[split] = arr
    return out


def _add_label_noise(payload: Dict[str, Any], sigma: float, rng) -> Dict[str, Any]:
    out = copy.deepcopy(payload)
    y = np.asarray(out["y_train"], dtype=float).copy()
    y += rng.normal(0.0, float(sigma), size=y.shape)
    out["y_train"] = np.clip(y, 0.0, 1.0)
    return out


def _skew_target(payload: Dict[str, Any], mode: str, rng) -> Dict[str, Any]:
    """Resample training rows so the target distribution becomes skewed."""
    out = copy.deepcopy(payload)
    y = np.asarray(out["y_train"], dtype=float)
    # Sampling weight falls off linearly toward the discouraged end.
    if mode == "low_heavy":
        weights = np.clip(1.0 - y, 1e-6, None)
    elif mode == "high_heavy":
        weights = np.clip(y, 1e-6, None)
    else:
        raise ValueError(f"Unknown target_skew mode {mode!r}.")
    weights = weights / weights.sum()
    idx = rng.choice(len(y), size=len(y), replace=True, p=weights)
    _take_train_rows(out, idx)
    return out


def _keep_top_features(
    payload: Dict[str, Any], fraction: float, rng, model=None,
) -> Dict[str, Any]:
    """Keep the most important logical features (whole one-hot groups).

    Importance is the increase in MAE when a feature's columns are permuted in
    the test split, measured with the baseline model.
    """
    out = copy.deepcopy(payload)
    names = list(out["feature_names"])
    numerical = list(out.get("numerical_features") or [])
    groups: Dict[str, List[int]] = {n: [names.index(n)] for n in numerical if n in names}
    for gname, meta in (out.get("categorical_groups") or {}).items():
        groups[gname] = [int(i) for i in meta["indices"]]

    X_test = np.asarray(out["X_test"], dtype=float)
    y_test = np.asarray(out["y_test"], dtype=float)
    base_pred = model.predict(X_test, verbose=0).reshape(-1)
    base_mae = float(np.mean(np.abs(base_pred - y_test)))
    importance: Dict[str, float] = {}
    for gname, cols in groups.items():
        permuted = X_test.copy()
        order = rng.permutation(len(permuted))
        permuted[:, cols] = permuted[order][:, cols]
        pred = model.predict(permuted, verbose=0).reshape(-1)
        importance[gname] = float(np.mean(np.abs(pred - y_test))) - base_mae

    keep_n = max(2, int(round(float(fraction) * len(groups))))
    keep = [g for g, _ in sorted(importance.items(), key=lambda kv: -kv[1])[:keep_n]]
    keep_cols = sorted(c for g in keep for c in groups[g])

    out["X_train"] = np.asarray(out["X_train"], dtype=float)[:, keep_cols]
    out["X_test"] = X_test[:, keep_cols]
    out["feature_names"] = [names[c] for c in keep_cols]
    remap = {old: new for new, old in enumerate(keep_cols)}
    out["numerical_features"] = [n for n in numerical if n in keep]
    out["categorical_groups"] = {
        g: {**meta, "indices": [remap[int(i)] for i in meta["indices"]]}
        for g, meta in (out.get("categorical_groups") or {}).items()
        if g in keep
    }
    out["dropped_features"] = [g for g in groups if g not in keep]
    out["feature_importance"] = importance
    out["raw_views_stale"] = True
    return out


def _numerical_only(payload: Dict[str, Any], rng) -> Dict[str, Any]:
    out = copy.deepcopy(payload)
    names = list(out["feature_names"])
    groups = dict(out.get("categorical_groups") or {})
    cat_cols = {int(i) for meta in groups.values() for i in meta["indices"]}
    keep_cols = [i for i in range(len(names)) if i not in cat_cols]
    out["X_train"] = np.asarray(out["X_train"], dtype=float)[:, keep_cols]
    out["X_test"] = np.asarray(out["X_test"], dtype=float)[:, keep_cols]
    out["feature_names"] = [names[c] for c in keep_cols]
    out["categorical_groups"] = {}
    out["dropped_features"] = sorted(groups)
    # Raw views still carry the dropped columns; keep them but flag the drift.
    out["raw_views_stale"] = True
    return out


def build_variant(
    dataset_key: str, factor: str, level: Any, *, seed: int, model=None,
) -> Optional[Dict[str, Any]]:
    """Return the variant payload, or ``None`` when the level is the baseline."""
    rng = np.random.default_rng(seed)
    payload = load_base(dataset_key)
    if factor == "train_size":
        if float(level) >= 1.0:
            return None
        out = _subsample_train(payload, float(level), rng)
    elif factor == "feature_noise":
        if float(level) <= 0.0:
            return None
        out = _add_feature_noise(payload, float(level), rng)
    elif factor == "label_noise":
        if float(level) <= 0.0:
            return None
        out = _add_label_noise(payload, float(level), rng)
    elif factor == "target_skew":
        if level == "none":
            return None
        out = _skew_target(payload, str(level), rng)
    elif factor == "n_features":
        if level == "all":
            return None
        out = _keep_top_features(payload, float(level), rng, model=model)
    elif factor == "numerical_only":
        if not level:
            return None
        if not payload.get("categorical_groups"):
            logger.info("[%s] has no categorical groups; numerical_only is a no-op.",
                        dataset_key)
            return None
        out = _numerical_only(payload, rng)
    else:
        raise ValueError(f"Unknown factor {factor!r}.")

    out["variant"] = {"base_dataset": dataset_key, "factor": factor,
                      "level": level, "seed": int(seed)}
    return out


def _write_priority_spec(dataset_key: str, key: str, payload: Dict[str, Any]) -> None:
    """Persist the variant's priority spec so the sets can be registered.

    The variant inherits the base dataset's spec, filtered to the features it
    still has; ``priority_sets.py`` picks these files up and generates the same
    loose/medium/strict family for the variant key.
    """
    spec = DATASET_SPECS.get(dataset_key)
    if spec is None:
        logger.warning("[%s] base dataset has no generic spec; the variant gets no "
                       "priority sets (hand-written sets are not transferable).", key)
        return
    numerical = {
        name: rule for name, rule in spec["numerical"].items()
        if name in set(payload.get("numerical_features") or [])
    }
    groups = payload.get("categorical_groups") or {}
    categorical = {
        name: [rule, [int(c) for c in groups[name]["categories"]]]
        for name, (rule, _codes) in spec["categorical"].items()
        if name in groups
    }
    out = {
        "base_dataset": dataset_key,
        "numerical": numerical,
        "categorical": categorical,
    }
    path = DATA_DIR / key / "priority_spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)


def materialise(
    dataset_key: str, factor: str, level: Any, *, seed: int, epochs: int,
    force: bool, model=None,
) -> Optional[str]:
    key = variant_key(dataset_key, factor, level, seed)
    data_path = DATA_DIR / key / "data.pkl"
    model_path = MODELS_DIR / key / "model.keras"
    if data_path.exists() and model_path.exists() and not force:
        logger.info("[%s] already materialised; skipping (use --force).", key)
        return key

    payload = build_variant(dataset_key, factor, level, seed=seed, model=model)
    if payload is None:
        return None
    payload["dataset_key"] = key

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as handle:
        pickle.dump(payload, handle)
    _write_priority_spec(dataset_key, key, payload)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    # Same architecture as the base dataset, so only the data differs.
    builder = MODEL_BUILDERS.get(dataset_key, build_default_regressor)
    variant_model = builder(payload["X_train"].shape[1])
    variant_model.fit(
        payload["X_train"], payload["y_train"],
        validation_split=0.1, epochs=int(epochs), batch_size=32, verbose=0,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    variant_model.save(model_path)

    pred = variant_model.predict(payload["X_test"], verbose=0).reshape(-1)
    y_test = np.asarray(payload["y_test"], dtype=float)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    logger.info("[%s] materialised: X_train=%s features=%d test_r2=%.3f",
                key, payload["X_train"].shape, payload["X_train"].shape[1], r2)
    return key


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="*", default=["auto_mpg", "carseats"])
    parser.add_argument("--factors", nargs="*", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="Print the variant keys that would be built and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("explainit").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    factors = list(FACTOR_LEVELS) if "all" in args.factors else list(args.factors)
    planned: List[Tuple[str, str, Any]] = [
        (dataset, factor, level)
        for dataset in args.datasets
        for factor in factors
        for level in FACTOR_LEVELS[factor]
    ]
    if args.list:
        for dataset, factor, level in planned:
            key = variant_key(dataset, factor, level, args.seed)
            baseline = level in (FACTOR_LEVELS[factor][0],)
            print(f"{'(baseline)' if baseline else key:45s} {dataset:12s} "
                  f"{factor:15s} {level}")
        return

    built: List[str] = []
    for dataset in args.datasets:
        base_model = None
        if "n_features" in factors:
            import tensorflow as tf

            base_model = tf.keras.models.load_model(MODELS_DIR / dataset / "model.keras")
        for factor in factors:
            for level in FACTOR_LEVELS[factor]:
                key = materialise(
                    dataset, factor, level, seed=args.seed, epochs=args.epochs,
                    force=args.force, model=base_model,
                )
                if key:
                    built.append(key)
    logger.info("\nMaterialised %d variant(s):\n  %s", len(built), "\n  ".join(built))


if __name__ == "__main__":
    main()
