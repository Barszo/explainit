"""Stage 1: prepare datasets for the continuous-target MINLP experiment.

For each registered dataset key this script:

* loads the raw features and target
* train/test-splits and standard-scales the features
* MinMax-scales the target into ``[0, 1]``
* pickles everything (plus the scalers) under ``data/<dataset_key>/data.pkl``

Adding a new dataset is a single line in :data:`DATASETS` plus the matching
loader function returning ``(X_df, y_array, feature_names)``.

CLI usage::

    python -m explainit.experiments.continuos_minlp.data_setup
    python -m explainit.experiments.continuos_minlp.data_setup --datasets diabetes
    python -m explainit.experiments.continuos_minlp.data_setup --force
"""

from __future__ import annotations

import argparse
import logging
import pickle
import ssl
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("explainit.experiments.continuos_minlp.data_setup")


# ---------------------------------------------------------------------------
# Raw loaders. Each returns (X dataframe, y array, feature_names list).
# ---------------------------------------------------------------------------


def _load_diabetes_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.astype(float).to_numpy()
    return X, y, list(bunch.feature_names)


def _load_california_housing_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from sklearn.datasets import fetch_california_housing

    try:
        bunch = fetch_california_housing(as_frame=True)
    except Exception as exc:  # pragma: no cover - macOS SSL fallback
        logger.warning(
            "fetch_california_housing failed (%s); retrying with SSL disabled.", exc,
        )
        ssl._create_default_https_context = ssl._create_unverified_context
        bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.astype(float).to_numpy()
    return X, y, list(bunch.feature_names)


def _load_synthetic_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    rng = np.random.default_rng(42)
    n_samples = 2000
    n_features = 6
    X_np = rng.normal(size=(n_samples, n_features))
    weights = np.linspace(1.0, 0.2, n_features)
    bias = 0.3
    y = X_np @ weights + bias + rng.normal(scale=0.1, size=n_samples)
    feature_names = [f"x{i}" for i in range(n_features)]
    X = pd.DataFrame(X_np, columns=feature_names)
    return X, y.astype(float), feature_names


# Registry. Add new datasets here.
DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, np.ndarray, List[str]]]] = {
    "diabetes": _load_diabetes_raw,
    # "california_housing": _load_california_housing_raw,
    # "synthetic": _load_synthetic_raw,
}


TARGET_NAMES: Dict[str, str] = {
    "diabetes": "disease_progression",
    "california_housing": "median_house_value",
    "synthetic": "synthetic_target",
}


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _prepare_dataset(
    key: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{key}'. Registered: {sorted(DATASETS)}"
        )

    X_raw, y_raw, feature_names = DATASETS[key]()
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

    return {
        "dataset_key": key,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "raw_target_min": float(np.min(y_raw)),
        "raw_target_max": float(np.max(y_raw)),
        "target_name": TARGET_NAMES.get(key, "target"),
    }


def _dataset_path(key: str) -> Path:
    return DATA_DIR / key / "data.pkl"


def setup_dataset(key: str, *, force: bool = False) -> Path:
    """Prepare and pickle a dataset, returning the path of the pickle file."""

    path = _dataset_path(key)
    if path.exists() and not force:
        logger.info("Dataset '%s' already cached at %s (use --force to recreate).", key, path)
        return path
    data = _prepare_dataset(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
    logger.info("Saved dataset '%s' to %s", key, path)
    return path


def load_dataset(key: str) -> Dict[str, object]:
    """Load a previously prepared dataset pickle."""

    path = _dataset_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached dataset for '{key}' at {path}. "
            f"Run: python -m explainit.experiments.continuos_minlp.data_setup "
            f"--datasets {key}"
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+", default=sorted(DATASETS),
        help="Dataset keys to prepare. Default: all registered.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-prepare even if a pickle already exists.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    for key in args.datasets:
        try:
            setup_dataset(key, force=args.force)
        except Exception as exc:
            logger.error("Failed to prepare '%s': %s", key, exc)


if __name__ == "__main__":
    main()
