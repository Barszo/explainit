"""Stage 2: train regression models for the continuous-target MINLP experiment.

For each dataset prepared by ``data_setup.py`` this script trains a
TensorFlow / Keras regression model and saves it under
``models/<dataset_key>/model.keras``.

Adding a new model means registering it in :data:`MODEL_BUILDERS`. The
default builder is a small two-layer MLP that matches the architecture
used in ``development/interactive_minlp_cont.py``.

CLI usage::

    python -m explainit.experiments.continuos_minlp.model_setup
    python -m explainit.experiments.continuos_minlp.model_setup --datasets diabetes --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402

from explainit.experiments.continuos_minlp.data_setup import (  # noqa: E402
    DATASETS,
    load_dataset,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("explainit.experiments.continuos_minlp.model_setup")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_default_regressor(input_dim: int) -> tf.keras.Model:
    """Default two-layer MLP regressor with ReLU hidden activations."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse",
    )
    return model


MODEL_BUILDERS: Dict[str, Callable[[int], tf.keras.Model]] = {
    "diabetes": build_default_regressor,
    # "california_housing": build_default_regressor,
    # "synthetic": build_default_regressor,
}


def _model_path(key: str) -> Path:
    return MODELS_DIR / key / "model.keras"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    key: str,
    *,
    epochs: int = 60,
    batch_size: int = 32,
    force: bool = False,
) -> Path:
    """Train and save a regression model for dataset ``key``."""

    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{key}'. Registered: {sorted(DATASETS)}"
        )
    builder = MODEL_BUILDERS.get(key, build_default_regressor)
    path = _model_path(key)
    if path.exists() and not force:
        logger.info(
            "Model for '%s' already cached at %s (use --force to retrain).",
            key, path,
        )
        return path

    data = load_dataset(key)
    X_train = np.asarray(
        data["X_train"].values if hasattr(data["X_train"], "values") else data["X_train"],
        dtype=float,
    )
    X_test = np.asarray(
        data["X_test"].values if hasattr(data["X_test"], "values") else data["X_test"],
        dtype=float,
    )
    y_train = np.asarray(data["y_train"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float)

    model = builder(X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=int(epochs), batch_size=int(batch_size), verbose=0,
    )
    train_loss = float(model.evaluate(X_train, y_train, verbose=0))
    test_loss = float(model.evaluate(X_test, y_test, verbose=0))
    logger.info(
        "Trained '%s' | scaled-y MSE train=%.4f test=%.4f",
        key, train_loss, test_loss,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("Saved model to %s", path)
    return path


def load_model(key: str) -> tf.keras.Model:
    path = _model_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached model for '{key}' at {path}. "
            f"Run: python -m explainit.experiments.continuos_minlp.model_setup "
            f"--datasets {key}"
        )
    return tf.keras.models.load_model(path)


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
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+", default=sorted(MODEL_BUILDERS),
        help="Dataset keys to train models for. Default: all registered.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    for key in args.datasets:
        try:
            train_model(
                key,
                epochs=args.epochs,
                batch_size=args.batch_size,
                force=args.force,
            )
        except Exception as exc:
            logger.error("Failed to train model for '%s': %s", key, exc)


if __name__ == "__main__":
    main()
