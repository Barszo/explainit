"""Stage 7: run random search as a baseline using the same priority sets.

Reads ``minlp_test_config.yaml`` and, for each (experiment * sample *
target) combination, invokes ``RandomSearchExplainer.generate_random_samples``
with the priority set selected from ``priority_sets.py``.

Output paths::

    results/<dataset_key>/<sample_idx>_<target>/random.json

The persisted metrics mirror ``minlp.json``: validity, time, iterations.
``iterations`` is the iteration count at which the best counterfactual
was found; the full per-CF iteration list is also recorded so finer
post-hoc analysis stays possible.

CLI usage::

    python -m explainit.experiments.continuos_minlp.random_runner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.explainers.random_search import RandomSearchExplainer  # noqa: E402

from explainit.experiments.continuous_minlp._context import load_context  # noqa: E402
from explainit.experiments.continuous_minlp.priority_sets import (  # noqa: E402
    ExperimentContext,
    build_priorities,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_CONFIG = EXPERIMENT_DIR / "minlp_test_config.yaml"

logger = logging.getLogger("explainit.experiments.continuos_minlp.random_runner")


def _model_predict_fn(model):
    def _predict(X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return model.predict(arr, verbose=0).flatten()
    return _predict


def _format_target(value: float) -> str:
    s = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _result_path(dataset_key: str, sample_idx: int, target_y: float, filename: str) -> Path:
    sub = f"{int(sample_idx)}_{_format_target(target_y)}"
    return RESULTS_DIR / dataset_key / sub / filename


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _merge_defaults(experiment: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(experiment)
    return merged


def run_random_pair(
    ctx: ExperimentContext,
    sample_idx: int,
    target_y: float,
    priority_set: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    sample = ctx.X_test[int(sample_idx)].astype(float)
    priorities = build_priorities(ctx, priority_set, sample)

    original_pred = float(_model_predict_fn(ctx.model)(sample.reshape(1, -1))[0])

    explainer = RandomSearchExplainer(
        model_pred=_model_predict_fn(ctx.model),
        priorities=priorities,
        sample=sample.tolist(),
        target=float(target_y),
    )

    started = time.perf_counter()
    error: Optional[str] = None
    samples_out: List[np.ndarray] = []
    preds_out: List[float] = []
    scores_out: List[float] = []
    iters_out: List[int] = []
    try:
        samples_out, preds_out, scores_out, iters_out = (
            explainer.generate_random_samples(
                expected_counterfactuals=int(settings.get("n_counterfactuals", 1)),
                max_iterations=int(settings.get("random_max_iterations", 10000)),
                epsilon=float(settings.get("epsilon", 0.05)),
                random_seed=settings.get("random_seed", None),
                use_monte_carlo=bool(settings.get("use_monte_carlo", True)),
            )
        )
    except Exception as exc:
        logger.exception(
            "Random search failed for sample=%d target=%.4f: %s",
            int(sample_idx), float(target_y), exc,
        )
        error = str(exc)
    elapsed = time.perf_counter() - started

    best_cf = (
        np.asarray(samples_out[0], dtype=float).flatten().tolist()
        if samples_out else None
    )
    best_pred = float(preds_out[0]) if preds_out else None
    best_iter = int(iters_out[0]) if iters_out else 0

    validity = bool(
        best_pred is not None
        and abs(best_pred - float(target_y)) <= float(settings.get("epsilon", 0.05))
    )

    return {
        "dataset": ctx.dataset_key,
        "priority_set": priority_set,
        "sample_index": int(sample_idx),
        "target_scaled": float(target_y),
        "epsilon": float(settings.get("epsilon", 0.05)),
        "original_prediction": original_pred,
        "counterfactual": best_cf,
        "counterfactual_prediction": best_pred,
        "preference_score": float(scores_out[0]) if scores_out else None,
        "validity": validity,
        "iterations": best_iter,
        "iterations_per_cf": [int(v) for v in iters_out],
        "n_counterfactuals_found": len(samples_out),
        "time_seconds": float(elapsed),
        "error": error,
    }


def run_experiment(
    experiment: Dict[str, Any],
    defaults: Dict[str, Any],
) -> List[Path]:
    settings = _merge_defaults(experiment, defaults)
    dataset_key = settings["dataset"]
    priority_set = settings.get("priority_set", "default")
    samples: Sequence[int] = settings.get("samples", [])
    targets: Sequence[float] = settings.get("targets", [])

    if not samples or not targets:
        logger.warning(
            "Experiment '%s' has no samples or targets; skipping.", dataset_key,
        )
        return []

    ctx = load_context(dataset_key)
    written: List[Path] = []
    for sample_idx in samples:
        if not (0 <= int(sample_idx) < len(ctx.X_test)):
            logger.warning("Sample index %s out of range; skipping.", sample_idx)
            continue
        for target_y in targets:
            result = run_random_pair(
                ctx, int(sample_idx), float(target_y), priority_set, settings,
            )
            path = _result_path(
                ctx.dataset_key, int(sample_idx), float(target_y), "random.json",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as handle:
                json.dump(result, handle, indent=2)
            written.append(path)
            logger.info(
                "[%s sample=%d target=%.4f] validity=%s iters=%d time=%.2fs -> %s",
                ctx.dataset_key, int(sample_idx), float(target_y),
                result["validity"], result["iterations"], result["time_seconds"],
                path,
            )
    return written


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG.name}).",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Optional filter -- run only experiments matching this dataset key.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    cfg = _load_config(Path(args.config))
    defaults = dict(cfg.get("defaults", {}) or {})
    experiments = list(cfg.get("experiments", []) or [])
    if args.dataset is not None:
        experiments = [e for e in experiments if e.get("dataset") == args.dataset]

    total = 0
    for exp in experiments:
        try:
            written = run_experiment(exp, defaults)
            total += len(written)
        except Exception as exc:
            logger.error("Experiment failed (%s): %s", exp.get("dataset"), exc)

    logger.info("Random runner done. Wrote %d result file(s).", total)


if __name__ == "__main__":
    main()
