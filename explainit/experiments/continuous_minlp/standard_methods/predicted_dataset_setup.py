"""Stage 4a: build a *predicted-target* dataset.

The counterfactual experiment targets the *model*, not the ground-truth
label. This stage loads the preprocessed dataset (``data_setup``) and the
trained model (``model_setup``), predicts the (scaled) target for every
row, and writes a new dataset pickle where ``y_train`` / ``y_test`` are the
**model predictions** instead of the original labels.

Everything else (feature matrix, scalers, categorical metadata) is carried
over unchanged, so the downstream selection / method-runner stages can load
this pickle exactly like the original one. The original labels are kept
under ``y_train_true`` / ``y_test_true`` for reference.

Output::

    standard_methods/predicted_data/<dataset_key>/data.pkl

CLI usage::

    python -m explainit.experiments.continuous_minlp.standard_methods.predicted_dataset_setup
    python -m explainit.experiments.continuous_minlp.standard_methods.predicted_dataset_setup --datasets diabetes --force
"""

from __future__ import annotations

import argparse
import copy
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.continuous_minlp.data_setup import (  # noqa: E402
    DATASETS,
    load_dataset,
)
from explainit.experiments.continuous_minlp.model_setup import load_model  # noqa: E402


STAGE_DIR = Path(__file__).resolve().parent
PREDICTED_DATA_DIR = STAGE_DIR / "predicted_data"
PREDICTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.standard_methods.predicted_dataset_setup"
)


def _as_matrix(arr) -> np.ndarray:
    return np.asarray(arr.values if hasattr(arr, "values") else arr, dtype=float)


def _predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0).reshape(-1)


def _predicted_path(key: str) -> Path:
    return PREDICTED_DATA_DIR / key / "data.pkl"


def build_predicted_dataset(key: str, *, force: bool = False) -> Path:
    """Create and pickle the predicted-target dataset for ``key``."""

    path = _predicted_path(key)
    if path.exists() and not force:
        logger.info(
            "[%s] SKIPPED: predicted dataset already exists at %s (pass --force).",
            key, path,
        )
        return path

    logger.info("[%s] Loading preprocessed dataset + trained model.", key)
    data = load_dataset(key)
    model = load_model(key)

    X_train = _as_matrix(data["X_train"])
    X_test = _as_matrix(data["X_test"])

    logger.info("[%s] Predicting scaled target for %d train / %d test rows.",
                key, len(X_train), len(X_test))
    y_train_pred = _predict(model, X_train)
    y_test_pred = _predict(model, X_test)

    new_data: Dict[str, object] = copy.deepcopy(data)
    # Keep the original labels for reference, then overwrite the target with
    # the model predictions so the rest of the pipeline "sees" the model.
    new_data["y_train_true"] = np.asarray(data["y_train"], dtype=float).reshape(-1)
    new_data["y_test_true"] = np.asarray(data["y_test"], dtype=float).reshape(-1)
    new_data["y_train"] = y_train_pred
    new_data["y_test"] = y_test_pred
    new_data["target_is_prediction"] = True

    # Predicted target in raw units (for plotting), if a y-scaler exists.
    y_scaler = data.get("y_scaler")
    if y_scaler is not None and hasattr(y_scaler, "inverse_transform"):
        new_data["y_train_pred_raw"] = (
            y_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).reshape(-1)
        )
        new_data["y_test_pred_raw"] = (
            y_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(-1)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(new_data, handle)

    logger.info(
        "[%s] DONE: saved predicted-target dataset to %s | "
        "pred target range train=[%.4f, %.4f] test=[%.4f, %.4f].",
        key, path,
        float(np.min(y_train_pred)), float(np.max(y_train_pred)),
        float(np.min(y_test_pred)), float(np.max(y_test_pred)),
    )
    return path


def load_predicted_dataset(key: str) -> Dict[str, object]:
    """Load a previously built predicted-target dataset pickle."""

    path = _predicted_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"No predicted dataset for '{key}' at {path}. Run: "
            f"python -m explainit.experiments.continuous_minlp.standard_methods."
            f"predicted_dataset_setup --datasets {key}"
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    for key in args.datasets:
        try:
            build_predicted_dataset(key, force=args.force)
        except Exception as exc:
            logger.error("Failed to build predicted dataset for '%s': %s", key, exc)


if __name__ == "__main__":
    main()
