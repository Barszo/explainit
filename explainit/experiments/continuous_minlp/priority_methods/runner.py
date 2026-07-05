"""Run the priority-branch methods (MINLP + random-search baseline).

For each experiment in ``config.yaml`` this:

1. loads the original dataset + model,
2. selects samples and derives targets (``prediction + target_offset``),
3. builds the sample-specific priorities from the selected priority set,
4. runs every configured method, and
5. writes linked result tables under ``results/<dataset_key>/``:

   * ``samples.csv``          -- one row per selected sample (``sample_id``),
   * ``counterfactuals.csv``  -- one row per counterfactual (adds
                                 ``priority_score``),
   * ``metrics_summary.csv``  -- per-method averages (adds
                                 ``avg_priority_score``),
   * ``summary.json``         -- machine-readable per-method summary,
   * ``run_config.json``      -- everything needed to interpret the run.

CLI usage::

    python -m explainit.experiments.continuous_minlp.priority_methods.runner
    python -m explainit.experiments.continuous_minlp.priority_methods.runner --dataset diabetes
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

from explainit.experiments.continuous_minlp.priority_sets import build_priorities  # noqa: E402
from explainit.experiments.continuous_minlp.priority_methods.methods import (  # noqa: E402
    build_method,
    compute_priority_score,
)
from explainit.experiments.continuous_minlp.priority_methods.selection import (  # noqa: E402
    PriorityContext,
    SampleRecord,
    load_priority_context,
    select_samples,
)

STAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STAGE_DIR / "results"
DEFAULT_CONFIG = STAGE_DIR / "config.yaml"
_CHANGE_TOL = 1e-6

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.priority_methods.runner"
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
    priorities: Dict[str, Any],
) -> Dict[str, Any]:
    if cf is None:
        return {
            "cf_prediction": None, "validity": False, "abs_pred_error": None,
            "l1": None, "l2": None, "n_changed": None, "sparsity_fraction": None,
            "priority_score": None,
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
        "priority_score": compute_priority_score(priorities, cf),
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _resolve_n_cfs(mc: Dict[str, Any], params: Dict[str, Any], default_n_cfs: int) -> int:
    if "n_cfs" in mc and mc["n_cfs"] is not None:
        return max(1, int(mc["n_cfs"]))
    if "n_cfs" in params and params["n_cfs"] is not None:
        return max(1, int(params.pop("n_cfs")))
    return max(1, int(default_n_cfs))


def _instantiate_methods(
    pctx: PriorityContext, method_cfgs: Sequence[Dict[str, Any]], epsilon: float,
    default_n_cfs: int,
) -> List[Any]:
    methods = []
    for mc in method_cfgs:
        name = mc["name"]
        params = dict(mc.get("params", {}) or {})
        n_cfs = _resolve_n_cfs(mc, params, default_n_cfs)
        try:
            method = build_method(name, pctx=pctx, epsilon=epsilon, **params)
            if not getattr(method, "supports_multiple", True):
                n_cfs = 1
            method._n_cfs = n_cfs
            methods.append(method)
        except Exception as exc:
            logger.error("Failed to instantiate method '%s': %s", name, exc)
    return methods


def _write_samples_csv(path: Path, pctx: PriorityContext, samples: List[SampleRecord]) -> None:
    fieldnames = ["sample_id", "original_prediction", "target"] + list(pctx.feature_names)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in samples:
            row = {
                "sample_id": rec.sample_id,
                "original_prediction": rec.original_prediction,
                "target": rec.target,
            }
            for i, name in enumerate(pctx.feature_names):
                row[name] = float(rec.x[i])
            writer.writerow(row)


def _write_counterfactuals_csv(path: Path, pctx: PriorityContext, rows: List[Dict[str, Any]]) -> None:
    metric_fields = [
        "sample_id", "method", "cf_index", "target", "original_prediction",
        "cf_prediction", "validity", "abs_pred_error", "l1", "l2", "n_changed",
        "sparsity_fraction", "priority_score", "iterations", "time_seconds", "error",
    ]
    fieldnames = metric_fields + [f"cf__{n}" for n in pctx.feature_names]
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
        "avg_sparsity_fraction", "avg_priority_score", "avg_iterations",
        "avg_time_seconds",
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
    pctx: PriorityContext,
    *,
    epsilon: float,
    priority_set: str,
    selection_cfg: Dict[str, Any],
    method_cfgs: Sequence[Dict[str, Any]],
    n_selected: int,
    n_cfs_by_method: Dict[str, int],
) -> None:
    payload = {
        "dataset": pctx.dataset_key,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epsilon": epsilon,
        "priority_set": priority_set,
        "n_samples_selected": int(n_selected),
        "selection": selection_cfg,
        "methods": [
            {"name": mc["name"],
             "n_cfs": n_cfs_by_method.get(mc["name"]),
             "params": {k: v for k, v in (mc.get("params", {}) or {}).items()
                        if k != "n_cfs"}}
            for mc in method_cfgs
        ],
        "feature_names": list(pctx.feature_names),
        "numerical_features": list(pctx.numerical_features),
        "categorical_groups": pctx.categorical_groups,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def run_experiment(experiment: Dict[str, Any], defaults: Dict[str, Any]) -> Optional[Path]:
    settings = _merge_defaults(experiment, defaults)
    dataset_key = settings["dataset"]
    priority_set = str(settings.get("priority_set", "set1"))
    epsilon = float(settings.get("epsilon", 0.05))
    selection_cfg = dict(settings.get("selection", {}) or {})
    method_cfgs = list(settings.get("methods", []) or [])

    pctx = load_priority_context(dataset_key)
    samples = select_samples(pctx, selection_cfg)
    if not samples:
        logger.warning("[%s] No samples selected; skipping.", dataset_key)
        return None
    logger.info("[%s] priority_set=%s | selected %d samples; running %d methods.",
                dataset_key, priority_set, len(samples), len(method_cfgs))

    default_n_cfs = int(settings.get("n_cfs", 1) or 1)
    methods = _instantiate_methods(pctx, method_cfgs, epsilon, default_n_cfs)

    cf_rows: List[Dict[str, Any]] = []
    per_method: Dict[str, List[Dict[str, Any]]] = {m.name: [] for m in methods}

    for rec in samples:
        priorities = build_priorities(pctx.ctx, priority_set, rec.x)
        for method in methods:
            n_cfs = int(getattr(method, "_n_cfs", 1))
            started = time.perf_counter()
            error: Optional[str] = None
            iterations: Optional[int] = None
            cf_iterations: Optional[List[int]] = None
            cfs: List[np.ndarray] = []
            try:
                out = method.generate_many(rec.x, rec.target, priorities, n_cfs)
                iterations = out.get("iterations")
                cf_iterations = out.get("cf_iterations")
                error = out.get("error")
                cfs = [np.asarray(c, dtype=float).reshape(-1)
                       for c in (out.get("cfs") or []) if c is not None]
            except Exception as exc:
                logger.exception("[%s] method=%s sample=%d failed: %s",
                                 dataset_key, method.name, rec.sample_id, exc)
                error = str(exc)
            elapsed = time.perf_counter() - started

            emitted = cfs if cfs else [None]
            n_valid = 0
            for cf_index, cf in enumerate(emitted):
                metrics = _compute_metrics(
                    rec.x, cf, rec.target, epsilon, pctx.model_predict, priorities)
                if metrics["validity"]:
                    n_valid += 1
                iters_for_cf = iterations
                if cf_iterations is not None and cf_index < len(cf_iterations):
                    iters_for_cf = cf_iterations[cf_index]
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
                    "priority_score": metrics["priority_score"],
                    "iterations": iters_for_cf,
                    "time_seconds": float(elapsed),
                    "error": error,
                }
                for i, name in enumerate(pctx.feature_names):
                    row[f"cf__{name}"] = float(cf[i]) if cf is not None else None
                cf_rows.append(row)
                per_method[method.name].append(row)

            logger.info(
                "[%s] sample=%d method=%s cfs=%d valid=%d iters=%s time=%.2fs",
                dataset_key, rec.sample_id, method.name, len(cfs), n_valid,
                iterations, elapsed,
            )

    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        name = method.name
        rows = per_method[name]
        # Group rows by sample to compute sample-level validity.
        by_sample: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_sample.setdefault(r["sample_id"], []).append(r)
        real_rows = [r for r in rows if r["cf_prediction"] is not None]
        valid_rows = [r for r in rows if r["validity"]]
        n_with_valid = sum(1 for rs in by_sample.values() if any(r["validity"] for r in rs))
        n_cfs_total = len(real_rows)
        n_cfs_valid = len(valid_rows)
        summary_rows.append({
            "dataset": dataset_key,
            "method": name,
            "n_cfs_requested": int(getattr(method, "_n_cfs", 1)),
            "n_samples": len(by_sample),
            "n_samples_with_valid": n_with_valid,
            "sample_validity_rate": (n_with_valid / len(by_sample)) if by_sample else 0.0,
            "n_cfs_total": int(n_cfs_total),
            "n_cfs_valid": int(n_cfs_valid),
            "cf_validity_rate": (n_cfs_valid / n_cfs_total) if n_cfs_total else 0.0,
            "avg_abs_pred_error": _mean([r["abs_pred_error"] for r in valid_rows]),
            "avg_l1": _mean([r["l1"] for r in valid_rows]),
            "avg_l2": _mean([r["l2"] for r in valid_rows]),
            "avg_n_changed": _mean([r["n_changed"] for r in valid_rows]),
            "avg_sparsity_fraction": _mean([r["sparsity_fraction"] for r in valid_rows]),
            "avg_priority_score": _mean([r["priority_score"] for r in valid_rows]),
            "avg_iterations": _mean([r["iterations"] for r in valid_rows]),
            "avg_time_seconds": _mean([r["time_seconds"] for r in real_rows or rows]),
        })

    out_dir = RESULTS_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_samples_csv(out_dir / "samples.csv", pctx, samples)
    _write_counterfactuals_csv(out_dir / "counterfactuals.csv", pctx, cf_rows)
    _write_summary(out_dir / "metrics_summary.csv", out_dir / "summary.json",
                   dataset_key, summary_rows)
    _write_run_config(
        out_dir / "run_config.json", pctx,
        epsilon=epsilon,
        priority_set=priority_set,
        selection_cfg=selection_cfg,
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
