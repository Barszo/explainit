"""Populate ``explainit/development/data`` and ``.../models`` for dev work.

The development directory is intended to be a self-contained sandbox that
does not depend on the experiment outputs at runtime. This setup script
fills the local caches once; everything else in ``development/`` then reads
from those caches.

Two sources are supported:

* ``--source binary_minlp`` (default) — copy the already-cached preprocessed
  datasets and trained Keras models from ``experiments/binary_minlp``. Fast,
  recommended.
* ``--source train`` — load datasets from scratch via the experiment's
  ``data_downloader``, train baseline Keras models via ``model_builder``,
  and save the artefacts into the development directory. Slow.

Examples::

    # Copy everything that exists in binary_minlp/
    python -m explainit.development.data_setup

    # Only copy german_credit + communities_crime
    python -m explainit.development.data_setup \\
        --source binary_minlp \\
        --datasets german_credit communities_crime

    # Overwrite existing files
    python -m explainit.development.data_setup --force

    # Re-train models from scratch (requires internet for downloads)
    python -m explainit.development.data_setup --source train \\
        --datasets german_credit --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEV_DIR = Path(__file__).resolve().parent
DEV_DATA_DIR = DEV_DIR / "data"
DEV_MODELS_DIR = DEV_DIR / "models"
DEV_DATA_DIR.mkdir(exist_ok=True)
DEV_MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENT_DIR = PROJECT_ROOT / "explainit" / "experiments" / "binary_minlp"

DATASET_FILES = {
    "communities_crime": "communities_crime_data.pkl",
    "german_credit": "german_credit_data.pkl",
    "credit_card_default": "credit_card_default_data.pkl",
    "lending_club": "lending_club_data.pkl",
}

MODEL_FILES = {
    "communities_crime": "communities_and_crime_model.keras",
    "german_credit": "german_credit_model.keras",
    "credit_card_default": "credit_card_default_model.keras",
    "lending_club": "lending_club_model.keras",
}

logger = logging.getLogger("explainit.development.data_setup")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Source: copy from experiments/binary_minlp
# ---------------------------------------------------------------------------


def _copy_file(src: Path, dst: Path, *, force: bool) -> bool:
    if not src.exists():
        logger.warning("Missing source file: %s", src)
        return False
    if dst.exists() and not force:
        logger.info("Already present (skip, use --force to overwrite): %s", dst)
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Copied %s -> %s", src, dst)
    return True


def copy_from_binary_minlp(datasets: Iterable[str], *, force: bool) -> List[str]:
    """Copy cached datasets and trained Keras models from binary_minlp."""
    populated: List[str] = []
    for key in datasets:
        if key not in DATASET_FILES:
            logger.warning("Unknown dataset key '%s'; skipping.", key)
            continue
        src_data = EXPERIMENT_DIR / "data" / DATASET_FILES[key]
        dst_data = DEV_DATA_DIR / DATASET_FILES[key]
        src_model = EXPERIMENT_DIR / "models" / MODEL_FILES[key]
        dst_model = DEV_MODELS_DIR / MODEL_FILES[key]
        ok_data = _copy_file(src_data, dst_data, force=force)
        ok_model = _copy_file(src_model, dst_model, force=force)
        if ok_data and ok_model:
            populated.append(key)
    return populated


# ---------------------------------------------------------------------------
# Source: train from scratch via the experiment's loaders/builder
# ---------------------------------------------------------------------------


def _import_experiment_helpers():
    """Import data_downloader + model_builder from the binary_minlp experiment.

    These modules live alongside the experiment script and are only needed
    for ``--source train``; the import is deferred so the copy path keeps
    working even if those modules are missing.
    """
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    import data_downloader  # type: ignore
    import model_builder  # type: ignore
    return data_downloader, model_builder


def train_from_scratch(
    datasets: Iterable[str],
    *,
    force: bool,
    epochs: int = 50,
    batch_size: int = 32,
) -> List[str]:
    """Load datasets and train fresh Keras models, saving into dev cache."""
    populated: List[str] = []
    data_downloader, model_builder = _import_experiment_helpers()

    loaders = {
        "communities_crime": getattr(data_downloader, "load_communities_and_crime", None),
        "german_credit": getattr(data_downloader, "load_german_credit", None),
        "credit_card_default": getattr(data_downloader, "load_credit_card_default", None),
        "lending_club": getattr(data_downloader, "load_lending_club_selected_features", None),
    }

    for key in datasets:
        if key not in DATASET_FILES:
            logger.warning("Unknown dataset key '%s'; skipping.", key)
            continue
        loader = loaders.get(key)
        if loader is None:
            logger.warning("No loader available for '%s'; skipping.", key)
            continue

        data_path = DEV_DATA_DIR / DATASET_FILES[key]
        model_path = DEV_MODELS_DIR / MODEL_FILES[key]
        if data_path.exists() and model_path.exists() and not force:
            logger.info("Cache already present for %s; skip (use --force).", key)
            populated.append(key)
            continue

        logger.info("Loading dataset: %s", key)
        if key == "lending_club":
            raw_csv = EXPERIMENT_DIR / "data" / "LoanStats3a.csv"
            X_train, X_test, y_train, y_test, feature_names, scaler = loader(str(raw_csv))
        else:
            X_train, X_test, y_train, y_test, feature_names, scaler = loader()

        if hasattr(feature_names, "tolist"):
            feature_names = feature_names.tolist()
        feature_names = list(feature_names)

        data_dict = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "feature_names": feature_names, "scaler": scaler,
            "train_shape": tuple(X_train.shape),
            "test_shape": tuple(X_test.shape),
            "n_features": X_train.shape[1],
        }
        with open(data_path, "wb") as handle:
            pickle.dump(data_dict, handle)
        logger.info("Saved dataset pickle: %s", data_path)

        logger.info("Training baseline model for %s (epochs=%d, batch=%d)...",
                    key, epochs, batch_size)
        model = model_builder.create_baseline_model(X_train.shape[1])
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size, verbose=0,
        )
        model.save(model_path)
        logger.info("Saved model: %s", model_path)
        populated.append(key)
    return populated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", choices=["binary_minlp", "train"], default="binary_minlp",
        help="Where to get the datasets/models from. Default: copy from binary_minlp.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=sorted(DATASET_FILES.keys()),
        help="Dataset keys to populate. Default: all known.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite files that already exist in development/data or /models.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs when --source train (default 50).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size when --source train (default 32).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    logger.info("Development directory: %s", DEV_DIR)
    logger.info("Data dir:   %s", DEV_DATA_DIR)
    logger.info("Models dir: %s", DEV_MODELS_DIR)

    if args.source == "binary_minlp":
        populated = copy_from_binary_minlp(args.datasets, force=args.force)
    else:
        populated = train_from_scratch(
            args.datasets, force=args.force,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    if populated:
        logger.info("Populated dev caches for: %s", ", ".join(populated))
    else:
        logger.warning("Nothing was populated. Check --source/--datasets/--force.")


if __name__ == "__main__":
    main()
