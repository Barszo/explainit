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
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from explainit.experiments.continuous_minlp.data_setup import (  # noqa: E402
    DATASETS,
    load_dataset,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ANALYSIS_DIR = EXPERIMENT_DIR / "model_analysis"
MODEL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("explainit.experiments.continuos_minlp.model_setup")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_mlp_regressor(
    input_dim: int,
    *,
    hidden_units: tuple[int, ...] = (64, 32),
    learning_rate: float = 1e-3,
    loss: str | tf.keras.losses.Loss = "mse",
    dropout: float = 0.0,
    l2: float = 0.0,
) -> tf.keras.Model:
    """Configurable MLP regressor for tabular regression."""
    regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units in hidden_units:
        model.add(
            tf.keras.layers.Dense(
                units, activation="relu", kernel_regularizer=regularizer,
            )
        )
        if dropout > 0.0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )
    return model


def build_default_regressor(input_dim: int) -> tf.keras.Model:
    return build_mlp_regressor(input_dim)


MODEL_BUILDERS: Dict[str, Callable[[int], tf.keras.Model]] = {
    "diabetes": lambda input_dim: build_mlp_regressor(
        input_dim,
        hidden_units=(192, 96, 48),
        learning_rate=7e-4,
        dropout=0.05,
        l2=1e-5,
    ),
    "diamonds": build_default_regressor,
    "bike_sharing_hourly": lambda input_dim: build_mlp_regressor(input_dim, hidden_units=(256, 128)),
    "insurance_charges": lambda input_dim: build_mlp_regressor(input_dim, hidden_units=(256, 128)),
    "wage": lambda input_dim: build_mlp_regressor(input_dim, dropout=0.1),
    "carseats": lambda input_dim: build_mlp_regressor(input_dim, hidden_units=(256, 128)),
    "cps85": lambda input_dim: build_mlp_regressor(
        input_dim,
        hidden_units=(256, 128),
        learning_rate=5e-4,
        loss=tf.keras.losses.Huber(),
        l2=1e-5,
    ),
    "auto_mpg": lambda input_dim: build_mlp_regressor(input_dim, l2=1e-4),
    "forest_fires": lambda input_dim: build_mlp_regressor(input_dim, hidden_units=()),
    "brazilian_houses_to_rent": lambda input_dim: build_mlp_regressor(input_dim, l2=1e-4),
    # "california_housing": build_default_regressor,
    # "synthetic": build_default_regressor,
}


TRAINING_DEFAULTS: Dict[str, Dict[str, int]] = {
    "diabetes": {"epochs": 200, "batch_size": 32},
    "diamonds": {"epochs": 120, "batch_size": 128},
    "bike_sharing_hourly": {"epochs": 120, "batch_size": 128},
    "insurance_charges": {"epochs": 200, "batch_size": 32},
    "wage": {"epochs": 200, "batch_size": 32},
    "carseats": {"epochs": 250, "batch_size": 32},
    "cps85": {"epochs": 250, "batch_size": 32},
    "auto_mpg": {"epochs": 250, "batch_size": 32},
    "forest_fires": {"epochs": 600, "batch_size": 32, "patience": 100},
    "brazilian_houses_to_rent": {"epochs": 200, "batch_size": 64},
}


def _model_path(key: str) -> Path:
    return MODELS_DIR / key / "model.keras"


def _model_analysis_path() -> Path:
    return MODEL_ANALYSIS_DIR / "model_analysis.csv"


def _compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if np.any(np.isclose(y_true, 0.0)):
        metrics["mape"] = float("nan")
    else:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    return metrics


def _metrics_to_str(metrics: Dict[str, float]) -> str:
    parts = [
        f"R2={metrics['r2']:.4f}",
        f"RMSE={metrics['rmse']:.4f}",
        f"MAE={metrics['mae']:.4f}",
        f"MSE={metrics['mse']:.4f}",
    ]
    mape = metrics.get("mape")
    if mape is not None:
        parts.append(
            "MAPE=nan (target contains zeros)"
            if np.isnan(mape) else f"MAPE={mape:.4f}"
        )
    return " | ".join(parts)


def _write_model_analysis_csv(
    reports: Sequence[Dict[str, Any]],
    failures: Sequence[tuple[str, str]],
) -> Path:
    path = _model_analysis_path()
    fieldnames = [
        "run_timestamp_utc",
        "status",
        "dataset_key",
        "target_name",
        "model_path",
        "scaled_test_r2",
        "scaled_test_rmse",
        "scaled_test_mae",
        "scaled_test_mse",
        "scaled_test_mape",
        "raw_test_r2",
        "raw_test_rmse",
        "raw_test_mae",
        "raw_test_mse",
        "raw_test_mape",
        "error",
    ]
    timestamp = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            scaled = report.get("metrics_test_scaled")
            raw = report.get("metrics_test_raw")
            writer.writerow({
                "run_timestamp_utc": timestamp,
                "status": report.get("status", "unknown"),
                "dataset_key": report.get("dataset_key", ""),
                "target_name": report.get("target_name", ""),
                "model_path": report.get("model_path", ""),
                "scaled_test_r2": scaled.get("r2") if isinstance(scaled, dict) else "",
                "scaled_test_rmse": scaled.get("rmse") if isinstance(scaled, dict) else "",
                "scaled_test_mae": scaled.get("mae") if isinstance(scaled, dict) else "",
                "scaled_test_mse": scaled.get("mse") if isinstance(scaled, dict) else "",
                "scaled_test_mape": scaled.get("mape") if isinstance(scaled, dict) else "",
                "raw_test_r2": raw.get("r2") if isinstance(raw, dict) else "",
                "raw_test_rmse": raw.get("rmse") if isinstance(raw, dict) else "",
                "raw_test_mae": raw.get("mae") if isinstance(raw, dict) else "",
                "raw_test_mse": raw.get("mse") if isinstance(raw, dict) else "",
                "raw_test_mape": raw.get("mape") if isinstance(raw, dict) else "",
                "error": "",
            })
        for dataset_key, error in failures:
            writer.writerow({
                "run_timestamp_utc": timestamp,
                "status": "failed",
                "dataset_key": dataset_key,
                "target_name": "",
                "model_path": "",
                "scaled_test_r2": "",
                "scaled_test_rmse": "",
                "scaled_test_mae": "",
                "scaled_test_mse": "",
                "scaled_test_mape": "",
                "raw_test_r2": "",
                "raw_test_rmse": "",
                "raw_test_mae": "",
                "raw_test_mse": "",
                "raw_test_mape": "",
                "error": error,
            })
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model_with_report(
    key: str,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a regression model for dataset ``key``."""
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{key}'. Registered: {sorted(DATASETS)}"
        )

    builder = MODEL_BUILDERS.get(key, build_default_regressor)
    path = _model_path(key)
    data = load_dataset(key)
    target_name = str(data.get("target_name", "target"))
    feature_names = list(data.get("feature_names", []))
    numerical_features = list(data.get("numerical_features", []))
    categorical_features = list(data.get("categorical_features", []))
    raw_target_min = data.get("raw_target_min")
    raw_target_max = data.get("raw_target_max")

    logger.info(
        "Dataset '%s' loaded | target=%s | train_rows=%d | test_rows=%d | features=%d "
        "(numerical=%d, categorical=%d) | raw_target_range=[%s, %s]",
        key,
        target_name,
        len(data["X_train"]),
        len(data["X_test"]),
        len(feature_names),
        len(numerical_features),
        len(categorical_features),
        f"{float(raw_target_min):.4f}" if raw_target_min is not None else "?",
        f"{float(raw_target_max):.4f}" if raw_target_max is not None else "?",
    )
    if path.exists() and not force:
        logger.info(
            "Model for '%s' already cached at %s (use --force to retrain).",
            key, path,
        )
        return {
            "dataset_key": key,
            "target_name": target_name,
            "status": "cached",
            "model_path": str(path),
            "metrics_test_scaled": None,
            "metrics_test_raw": None,
        }

    defaults = TRAINING_DEFAULTS.get(key, {})
    resolved_epochs = int(epochs if epochs is not None else defaults.get("epochs", 200))
    resolved_batch_size = int(batch_size if batch_size is not None else defaults.get("batch_size", 32))
    resolved_patience = int(defaults.get("patience", 25))

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
    train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    test_mask = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
    if not np.all(train_mask) or not np.all(test_mask):
        logger.warning(
            "Dataset '%s' contains non-finite values, dropping train=%d test=%d rows before training.",
            key,
            int((~train_mask).sum()),
            int((~test_mask).sum()),
        )
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = builder(X_train.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=resolved_patience, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(10, resolved_patience // 2),
            min_lr=1e-5,
        ),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    y_pred_train = model.predict(X_train, verbose=0).reshape(-1)
    y_pred_test = model.predict(X_test, verbose=0).reshape(-1)
    metrics_train_scaled = _compute_regression_metrics(y_train, y_pred_train)
    metrics_test_scaled = _compute_regression_metrics(y_test, y_pred_test)

    metrics_train_raw: Optional[Dict[str, float]] = None
    metrics_test_raw: Optional[Dict[str, float]] = None
    y_scaler = data.get("y_scaler")
    if y_scaler is not None and hasattr(y_scaler, "inverse_transform"):
        y_train_raw = y_scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test_raw = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_pred_train_raw = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).reshape(-1)
        y_pred_test_raw = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).reshape(-1)
        metrics_train_raw = _compute_regression_metrics(y_train_raw, y_pred_train_raw)
        metrics_test_raw = _compute_regression_metrics(y_test_raw, y_pred_test_raw)

    logger.info(
        "Trained '%s' | scaled-train: %s",
        key, _metrics_to_str(metrics_train_scaled),
    )
    logger.info(
        "Trained '%s' | scaled-test:  %s",
        key, _metrics_to_str(metrics_test_scaled),
    )
    if metrics_train_raw is not None and metrics_test_raw is not None:
        logger.info(
            "Trained '%s' | raw-train:    %s",
            key, _metrics_to_str(metrics_train_raw),
        )
        logger.info(
            "Trained '%s' | raw-test:     %s",
            key, _metrics_to_str(metrics_test_raw),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("Saved model to %s", path)
    return {
        "dataset_key": key,
        "target_name": target_name,
        "status": "trained",
        "model_path": str(path),
        "metrics_test_scaled": metrics_test_scaled,
        "metrics_test_raw": metrics_test_raw,
    }


def train_model(
    key: str,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    force: bool = False,
) -> Path:
    """Train and save a regression model for dataset ``key``."""
    report = train_model_with_report(
        key,
        epochs=epochs,
        batch_size=batch_size,
        force=force,
    )
    return Path(str(report["model_path"]))


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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    reports: list[Dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for key in args.datasets:
        try:
            report = train_model_with_report(
                key,
                epochs=args.epochs,
                batch_size=args.batch_size,
                force=args.force,
            )
            reports.append(report)
        except Exception as exc:
            logger.error("Failed to train model for '%s': %s", key, exc)
            failures.append((key, str(exc)))

    logger.info(
        "Training summary | success=%d | failed=%d",
        len(reports),
        len(failures),
    )
    for report in reports:
        test_metrics = report.get("metrics_test_raw") or report.get("metrics_test_scaled")
        test_metrics_str = (
            _metrics_to_str(test_metrics)
            if isinstance(test_metrics, dict) else "n/a (cached model not retrained)"
        )
        logger.info(
            "Summary item | status=%s | dataset=%s | target=%s | model_path=%s | test_metrics=%s",
            report.get("status"),
            report.get("dataset_key"),
            report.get("target_name"),
            report.get("model_path"),
            test_metrics_str,
        )
    for key, error in failures:
        logger.info("Summary item | status=failed | dataset=%s | error=%s", key, error)
    analysis_path = _write_model_analysis_csv(reports, failures)
    logger.info("Saved training metrics summary CSV to %s", analysis_path)


if __name__ == "__main__":
    main()
