"""Execute a single :class:`RunSpec`.

One run = one method, one parameter configuration, one seed, over all samples
selected by the spec. Everything method-specific is delegated to the
``continuous_minlp.priority_methods`` package, so the paper studies and the dev
pipeline compute identical metrics.

Two properties matter for the paper and are enforced here:

* **priority isolation** -- the MINLP search mutates the priority dict it is
  given (Stage 0 derives bounds/allowed intervals in place), so every method
  run gets a freshly built dict and scoring uses a pristine one;
* **budget accounting** -- the method only ever sees a counting proxy of the
  model, so ``model_rows`` is a fair cost axis across methods.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from explainit.experiments.continuous_minlp.priority_sets import build_priorities
from explainit.experiments.continuous_minlp.priority_methods.methods import (
    max_attainable_priority,
)
from explainit.experiments.paper.common.standard_bridge import build_any_method
from explainit.experiments.continuous_minlp.priority_methods.metrics import (
    compute_cf_metrics,
)
from explainit.experiments.continuous_minlp.priority_methods.selection import (
    BudgetExceeded,
    PriorityContext,
    counted_context,
    load_priority_context,
    select_samples,
)
from explainit.experiments.paper.common.runspec import RunSpec

logger = logging.getLogger("explainit.experiments.paper.execute")

_CONTEXT_CACHE: Dict[str, PriorityContext] = {}

_EXTRA_FIELDS = (
    "reached_target", "stop_reason", "cf_source", "exemplar_source",
    "exemplar_pred_distance", "anchor_priority_score", "priority_gain_vs_anchor",
    "peak_lock_in_moves", "peak_lock_in_gain", "peak_lock_in_model_calls",
    "warm_start_total_combos", "warm_start_feasible_combos",
    "warm_start_best_model_gap", "accepted_steps", "rejected_steps",
    "restoration_hits", "trust_fraction_final", "search_exception",
)


def get_context(dataset_key: str) -> PriorityContext:
    """Load (and cache) the dataset + model context for ``dataset_key``."""
    if dataset_key not in _CONTEXT_CACHE:
        logger.info("Loading context for dataset '%s'.", dataset_key)
        _CONTEXT_CACHE[dataset_key] = load_priority_context(dataset_key)
    return _CONTEXT_CACHE[dataset_key]


def _mean(values) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def execute_run(spec: RunSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run one spec; return ``(cf_rows, sample_rows, summary)``."""

    pctx = get_context(spec.dataset)
    samples = select_samples(pctx, dict(spec.selection))
    if not samples:
        raise ValueError(
            f"Run {spec.run_id}: no samples satisfy the selection block {spec.selection}."
        )

    params = dict(spec.params)
    params["seed"] = spec.seed  # the spec's seed always wins
    # A budget caps model evaluations so methods can be compared at equal cost;
    # exceeding it aborts that sample and is recorded as `budget_exceeded`.
    max_model_rows = params.pop("max_model_rows", None)
    counted_pctx, counter = counted_context(
        pctx, max_rows=None if max_model_rows is None else int(max_model_rows))
    method = build_any_method(
        spec.method, pctx=counted_pctx, epsilon=spec.epsilon, **params)
    n_cfs = 1 if not getattr(method, "supports_multiple", True) else int(spec.n_cfs)

    cf_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    started_run = time.perf_counter()

    for position, rec in enumerate(samples, start=1):
        scoring_priorities = build_priorities(pctx.ctx, spec.priority_set, rec.x)
        max_priority = max_attainable_priority(scoring_priorities)
        sample_row = {
            "sample_id": rec.sample_id,
            "original_prediction": rec.original_prediction,
            "target": rec.target,
            "max_attainable_priority": max_priority,
        }
        for i, name in enumerate(pctx.feature_names):
            sample_row[name] = float(rec.x[i])
        sample_rows.append(sample_row)

        counter.reset()
        started = time.perf_counter()
        error: Optional[str] = None
        cfs: List[np.ndarray] = []
        extra: Dict[str, Any] = {}
        iterations: Optional[int] = None
        cf_iterations: Optional[List[int]] = None
        try:
            out = method.generate_many(
                rec.x, rec.target,
                build_priorities(pctx.ctx, spec.priority_set, rec.x),
                n_cfs,
            )
            iterations = out.get("iterations")
            cf_iterations = out.get("cf_iterations")
            error = out.get("error")
            extra = dict(out.get("extra", {}) or {})
            cfs = [np.asarray(c, dtype=float).reshape(-1)
                   for c in (out.get("cfs") or []) if c is not None]
        except BudgetExceeded as exc:
            logger.info("[%s] sample=%d hit the budget: %s", spec.run_id, rec.sample_id, exc)
            error = "budget_exceeded"
        except Exception as exc:  # a failed cell must not abort the sweep
            logger.warning("[%s] sample=%d failed: %s", spec.run_id, rec.sample_id, exc)
            error = str(exc)
        elapsed = time.perf_counter() - started

        for cf_index, cf in enumerate(cfs or [None]):
            metrics = compute_cf_metrics(
                rec.x, cf, rec.target, spec.epsilon, pctx.model_predict,
                scoring_priorities, max_priority)
            iters = iterations
            if cf_iterations is not None and cf_index < len(cf_iterations):
                iters = cf_iterations[cf_index]
            row: Dict[str, Any] = {
                "sample_id": rec.sample_id,
                "cf_index": cf_index,
                "target": rec.target,
                "original_prediction": rec.original_prediction,
                "max_attainable_priority": max_priority,
                **metrics,
                "iterations": iters,
                "time_seconds": float(elapsed),
                "model_calls": int(counter.n_calls),
                "model_rows": int(counter.n_rows),
                "error": error,
            }
            for key in _EXTRA_FIELDS:
                row[key] = extra.get(key)
            for i, name in enumerate(pctx.feature_names):
                row[f"cf__{name}"] = float(cf[i]) if cf is not None else None
            cf_rows.append(row)

        logger.info(
            "[%s] sample %d/%d (id=%d): %d cf(s), %d valid, %d model rows, %.2fs",
            spec.run_id, position, len(samples), rec.sample_id, len(cfs),
            sum(1 for r in cf_rows[-max(1, len(cfs)):] if r["validity"]),
            counter.n_rows, elapsed,
        )

    valid = [r for r in cf_rows if r["validity"]]
    produced = [r for r in cf_rows if r["cf_prediction"] is not None]
    by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for row in cf_rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    n_with_valid = sum(1 for rows in by_sample.values() if any(r["validity"] for r in rows))

    summary = {
        "run_id": spec.run_id,
        "n_samples": len(samples),
        "n_samples_with_valid": n_with_valid,
        "sample_validity_rate": (n_with_valid / len(samples)) if samples else 0.0,
        "n_cfs_total": len(produced),
        "n_cfs_valid": len(valid),
        "cf_validity_rate": (len(valid) / len(produced)) if produced else 0.0,
        "avg_abs_pred_error": _mean([r["abs_pred_error"] for r in valid]),
        "avg_l1": _mean([r["l1"] for r in valid]),
        "avg_l2": _mean([r["l2"] for r in valid]),
        "avg_n_changed": _mean([r["n_changed"] for r in valid]),
        "avg_sparsity_fraction": _mean([r["sparsity_fraction"] for r in valid]),
        "avg_priority_score": _mean([r["priority_score"] for r in valid]),
        "avg_priority_score_normalised": _mean(
            [r["priority_score_normalised"] for r in valid]),
        "avg_iterations": _mean([r["iterations"] for r in valid]),
        "avg_time_seconds": _mean([r["time_seconds"] for r in cf_rows]),
        "avg_model_rows": _mean([r["model_rows"] for r in cf_rows]),
        "avg_model_calls": _mean([r["model_calls"] for r in cf_rows]),
        "total_wall_seconds": float(time.perf_counter() - started_run),
        "n_errors": sum(1 for r in cf_rows if r["error"]),
        "n_budget_exceeded": sum(1 for r in cf_rows if r["error"] == "budget_exceeded"),
        "max_model_rows": max_model_rows,
    }
    return cf_rows, sample_rows, summary


__all__ = ["execute_run", "get_context"]
