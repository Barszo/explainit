"""Stage 4b: run the standard regression counterfactual methods.

For each experiment in ``config.yaml`` this:

1. loads the predicted-target dataset + model,
2. selects samples and derives targets,
3. builds per-sample actionability (bounds + immutable features),
4. runs every configured method, and
5. writes linked result tables under ``results/<dataset_key>/``:

   * ``samples.csv``          -- one row per selected sample (``sample_id``),
   * ``counterfactuals.csv``  -- one row per generated counterfactual
                                 (``sample_id``, ``method``, ``cf_index``),
   * ``metrics_summary.csv``  -- per-method average metrics,
   * ``summary.json``         -- machine-readable summary.

The number of counterfactuals requested per method is controlled by ``n_cfs``
(per-method key, or the ``defaults.n_cfs`` fallback). Methods that cannot
produce multiple distinct counterfactuals (``supports_multiple == False``) are
clamped to a single counterfactual.

CLI usage::

    python -m explainit.experiments.continuous_minlp.standard_methods.runner
    python -m explainit.experiments.continuous_minlp.standard_methods.runner --dataset diabetes
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.continuous_minlp.standard_methods.methods import (  # noqa: E402
    build_method,
)
from explainit.experiments.continuous_minlp.standard_methods.selection import (  # noqa: E402
    PredictedContext,
    SampleRecord,
    build_actionability,
    load_predicted_context,
    select_samples,
)

STAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STAGE_DIR / "results"
DEFAULT_CONFIG = STAGE_DIR / "config.yaml"
_CHANGE_TOL = 1e-6

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.standard_methods.runner"
)


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _merge_defaults(experiment: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(experiment)
    return merged


def _compute_metrics(
    x: np.ndarray,
    cf: Optional[np.ndarray],
    target: float,
    epsilon: float,
    model_predict,
) -> Dict[str, Any]:
    if cf is None:
        return {
            "cf_prediction": None,
            "validity": False,
            "abs_pred_error": None,
            "l1": None,
            "l2": None,
            "n_changed": None,
            "sparsity_fraction": None,
        }
    cf = np.asarray(cf, dtype=float)
    cf_pred = float(model_predict(cf.reshape(1, -1))[0])
    abs_err = abs(cf_pred - float(target))
    diff = np.abs(cf - x)
    n_changed = int(np.sum(diff > _CHANGE_TOL))
    return {
        "cf_prediction": cf_pred,
        "validity": bool(abs_err <= epsilon),
        "abs_pred_error": abs_err,
        "l1": float(np.sum(diff)),
        "l2": float(np.linalg.norm(cf - x)),
        "n_changed": n_changed,
        "sparsity_fraction": float(n_changed / len(x)),
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _resolve_n_cfs(mc: Dict[str, Any], params: Dict[str, Any], default_n_cfs: int) -> int:
    """Number of counterfactuals to request for a method entry.

    Looks first at the method-level ``n_cfs`` key, then at ``params.n_cfs``,
    then at the global default. ``params`` is mutated to drop ``n_cfs`` so it
    is not forwarded to the method constructor.
    """
    if "n_cfs" in mc and mc["n_cfs"] is not None:
        return max(1, int(mc["n_cfs"]))
    if "n_cfs" in params and params["n_cfs"] is not None:
        return max(1, int(params.pop("n_cfs")))
    return max(1, int(default_n_cfs))


def _instantiate_methods(
    ctx: PredictedContext, method_cfgs: Sequence[Dict[str, Any]], epsilon: float,
    default_n_cfs: int,
) -> List[Any]:
    methods = []
    for mc in method_cfgs:
        name = mc["name"]
        params = dict(mc.get("params", {}) or {})
        n_cfs = _resolve_n_cfs(mc, params, default_n_cfs)
        try:
            method = build_method(
                name,
                model=ctx.model,
                model_predict=ctx.model_predict,
                X_train=ctx.X_train,
                feature_names=ctx.feature_names,
                epsilon=epsilon,
                y_train=ctx.y_train,
                **params,
            )
            if not getattr(method, "supports_multiple", True):
                n_cfs = 1
            method._n_cfs = n_cfs  # requested count for this run
            methods.append(method)
        except Exception as exc:
            logger.error("Failed to instantiate method '%s': %s", name, exc)
    return methods


def _write_samples_csv(path: Path, ctx: PredictedContext, samples: List[SampleRecord]) -> None:
    fieldnames = ["sample_id", "original_prediction", "target"] + list(ctx.feature_names)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in samples:
            row = {
                "sample_id": rec.sample_id,
                "original_prediction": rec.original_prediction,
                "target": rec.target,
            }
            for i, name in enumerate(ctx.feature_names):
                row[name] = float(rec.x[i])
            writer.writerow(row)


def _write_counterfactuals_csv(path: Path, ctx: PredictedContext, rows: List[Dict[str, Any]]) -> None:
    metric_fields = [
        "sample_id", "method", "cf_index", "target", "original_prediction",
        "cf_prediction", "validity", "abs_pred_error", "l1", "l2", "n_changed",
        "sparsity_fraction", "iterations", "time_seconds", "error",
    ]
    fieldnames = metric_fields + [f"cf__{n}" for n in ctx.feature_names]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(path_csv: Path, path_json: Path, dataset_key: str,
                   summary_rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset", "method", "n_cfs_requested", "n_samples",
        "n_samples_with_valid", "sample_validity_rate",
        "n_cfs_total", "n_cfs_valid", "cf_validity_rate",
        "avg_abs_pred_error", "avg_l1", "avg_l2", "avg_n_changed",
        "avg_sparsity_fraction", "avg_iterations", "avg_time_seconds",
    ]
    with open(path_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    with open(path_json, "w", encoding="utf-8") as handle:
        json.dump({"dataset": dataset_key, "methods": summary_rows}, handle, indent=2)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_run_config(
    path: Path,
    ctx: PredictedContext,
    *,
    epsilon: float,
    selection_cfg: Dict[str, Any],
    actionability_cfg: Dict[str, Any],
    method_cfgs: Sequence[Dict[str, Any]],
    n_selected: int,
    n_cfs_by_method: Dict[str, int],
) -> None:
    """Persist everything needed to interpret a run: the exact parameters,
    selection / actionability config, and the feature layout.
    """
    payload = {
        "dataset": ctx.dataset_key,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epsilon": epsilon,
        "n_samples_selected": int(n_selected),
        "selection": selection_cfg,
        "actionability": actionability_cfg,
        "methods": [
            {"name": mc["name"],
             "n_cfs": n_cfs_by_method.get(mc["name"]),
             "params": {k: v for k, v in (mc.get("params", {}) or {}).items()
                        if k != "n_cfs"}}
            for mc in method_cfgs
        ],
        "feature_names": list(ctx.feature_names),
        "numerical_features": list(ctx.numerical_features),
        "categorical_groups": ctx.categorical_groups,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def run_experiment(experiment: Dict[str, Any], defaults: Dict[str, Any]) -> Optional[Path]:
    settings = _merge_defaults(experiment, defaults)
    dataset_key = settings["dataset"]
    epsilon = float(settings.get("epsilon", 0.05))
    selection_cfg = dict(settings.get("selection", {}) or {})
    actionability_cfg = dict(settings.get("actionability", {}) or {})
    method_cfgs = list(settings.get("methods", []) or [])

    ctx = load_predicted_context(dataset_key)
    samples = select_samples(ctx, selection_cfg)
    if not samples:
        logger.warning("[%s] No samples selected; skipping.", dataset_key)
        return None
    logger.info("[%s] Selected %d samples; running %d methods.",
                dataset_key, len(samples), len(method_cfgs))

    default_n_cfs = int(settings.get("n_cfs", 1) or 1)
    methods = _instantiate_methods(ctx, method_cfgs, epsilon, default_n_cfs)

    cf_rows: List[Dict[str, Any]] = []
    per_method: Dict[str, List[Dict[str, Any]]] = {m.name: [] for m in methods}
    # Per (sample, method) call-level stats, keyed by method name.
    per_call: Dict[str, List[Dict[str, Any]]] = {m.name: [] for m in methods}

    for rec in samples:
        bounds, features_to_vary = build_actionability(ctx, actionability_cfg, rec.x)
        for method in methods:
            n_cfs = int(getattr(method, "_n_cfs", 1))
            started = time.perf_counter()
            error: Optional[str] = None
            iterations: Optional[int] = None
            cfs: List[np.ndarray] = []
            try:
                out = method.generate_many(
                    rec.x, rec.target, bounds, features_to_vary, n_cfs)
                iterations = out.get("iterations")
                error = out.get("error")
                cfs = [np.asarray(c, dtype=float).reshape(-1)
                       for c in (out.get("cfs") or []) if c is not None]
            except Exception as exc:
                logger.exception("[%s] method=%s sample=%d failed: %s",
                                 dataset_key, method.name, rec.sample_id, exc)
                error = str(exc)
            elapsed = time.perf_counter() - started

            # Always emit at least one row so every (sample, method) is visible.
            emitted = cfs if cfs else [None]
            n_valid = 0
            for cf_index, cf in enumerate(emitted):
                metrics = _compute_metrics(rec.x, cf, rec.target, epsilon, ctx.model_predict)
                if metrics["validity"]:
                    n_valid += 1
                row: Dict[str, Any] = {
                    "sample_id": rec.sample_id,
                    "method": method.name,
                    "cf_index": cf_index,
                    "target": rec.target,
                    "original_prediction": rec.original_prediction,
                    "cf_prediction": metrics["cf_prediction"],
                    "validity": metrics["validity"],
                    "abs_pred_error": metrics["abs_pred_error"],
                    "l1": metrics["l1"],
                    "l2": metrics["l2"],
                    "n_changed": metrics["n_changed"],
                    "sparsity_fraction": metrics["sparsity_fraction"],
                    "iterations": iterations,
                    "time_seconds": float(elapsed),
                    "error": error,
                }
                for i, name in enumerate(ctx.feature_names):
                    row[f"cf__{name}"] = float(cf[i]) if cf is not None else None
                cf_rows.append(row)
                per_method[method.name].append(row)

            per_call[method.name].append({
                "sample_id": rec.sample_id,
                "n_returned": len(cfs),
                "n_valid": n_valid,
                "iterations": iterations,
                "time_seconds": float(elapsed),
            })
            logger.info(
                "[%s] sample=%d method=%s cfs=%d valid=%d iters=%s time=%.2fs",
                dataset_key, rec.sample_id, method.name, len(cfs), n_valid,
                iterations, elapsed,
            )

    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        name = method.name
        rows = per_method[name]
        calls = per_call[name]
        valid_rows = [r for r in rows if r["validity"]]
        n_cfs_total = sum(c["n_returned"] for c in calls)
        n_cfs_valid = sum(c["n_valid"] for c in calls)
        n_with_valid = sum(1 for c in calls if c["n_valid"] > 0)
        summary_rows.append({
            "dataset": dataset_key,
            "method": name,
            "n_cfs_requested": int(getattr(method, "_n_cfs", 1)),
            "n_samples": len(calls),
            "n_samples_with_valid": n_with_valid,
            "sample_validity_rate": (n_with_valid / len(calls)) if calls else 0.0,
            "n_cfs_total": int(n_cfs_total),
            "n_cfs_valid": int(n_cfs_valid),
            "cf_validity_rate": (n_cfs_valid / n_cfs_total) if n_cfs_total else 0.0,
            "avg_abs_pred_error": _mean([r["abs_pred_error"] for r in valid_rows]),
            "avg_l1": _mean([r["l1"] for r in valid_rows]),
            "avg_l2": _mean([r["l2"] for r in valid_rows]),
            "avg_n_changed": _mean([r["n_changed"] for r in valid_rows]),
            "avg_sparsity_fraction": _mean([r["sparsity_fraction"] for r in valid_rows]),
            "avg_iterations": _mean([c["iterations"] for c in calls]),
            "avg_time_seconds": _mean([c["time_seconds"] for c in calls]),
        })

    out_dir = RESULTS_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_samples_csv(out_dir / "samples.csv", ctx, samples)
    _write_counterfactuals_csv(out_dir / "counterfactuals.csv", ctx, cf_rows)
    _write_summary(out_dir / "metrics_summary.csv", out_dir / "summary.json",
                   dataset_key, summary_rows)
    _write_run_config(
        out_dir / "run_config.json", ctx,
        epsilon=epsilon,
        selection_cfg=selection_cfg,
        actionability_cfg=actionability_cfg,
        method_cfgs=method_cfgs,
        n_selected=len(samples),
        n_cfs_by_method={m.name: int(getattr(m, "_n_cfs", 1)) for m in methods},
    )
    logger.info("[%s] Wrote results to %s", dataset_key, out_dir)
    return out_dir


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
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=None,
                        help="Optional filter -- run only this dataset key.")
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

    for exp in experiments:
        try:
            run_experiment(exp, defaults)
        except Exception as exc:
            logger.error("Experiment failed (%s): %s", exp.get("dataset"), exc)


if __name__ == "__main__":
    main()
