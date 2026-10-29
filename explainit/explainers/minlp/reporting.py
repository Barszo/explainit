"""Progress reporting for the MINLP search."""

import numpy as np

from explainit.explainers.minlp.intervals import in_allowed_intervals
from explainit.logging_config import logger


def log_bounds(priorities_state):
    """Surface the numerical bounds used by both LP and SLSQP."""
    logger.info("--- STAGE 2/6: Collect numerical-feature bounds from priorities ---")
    logger.info(
        "These (min, max) pairs constrain how far each numerical feature is "
        "allowed to move during the LP and SLSQP searches."
    )
    logger.info("Bounds for numerical features: %s", priorities_state.bounds)
    logger.info(
        "Non-actionable feature indices (frozen at sample values): %s",
        priorities_state.non_actionable_indices,
    )


def log_workflow_summary(
    workflow_log, model, sample, target, dataset, priorities_state, feature_label
):
    """Write the initial prediction and per-feature bounds coverage to ``workflow_log``."""
    initial_prediction = float(model([sample])[0])
    workflow_log("")
    workflow_log("==== MINLP WORKFLOW SUMMARY ====")
    workflow_log("1) Initial prediction: %.6f", initial_prediction)
    workflow_log("2) Target: %.6f", float(target))
    workflow_log("")
    workflow_log("3) Bounds per feature vs dataset min/max:")
    n_rows = int(dataset.shape[0])
    global_allowed_mask = np.ones(n_rows, dtype=bool)

    for idx in sorted(priorities_state.numerical_priorities.keys()):
        dataset_min = float(np.min(dataset[:, idx]))
        dataset_max = float(np.max(dataset[:, idx]))
        cfg = priorities_state.numerical_priorities.get(idx, {})
        bounds_min, bounds_max = priorities_state.bounds[idx]
        allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None

        allowed_min = dataset_min if bounds_min is None else float(bounds_min)
        allowed_max = dataset_max if bounds_max is None else float(bounds_max)
        dataset_span = dataset_max - dataset_min
        allowed_span = allowed_max - allowed_min
        coverage_pct = 100.0 if abs(dataset_span) < 1e-12 else (allowed_span / dataset_span) * 100.0
        if allowed_intervals:
            feature_allowed_mask = np.array(
                [in_allowed_intervals(float(v), allowed_intervals) for v in dataset[:, idx]],
                dtype=bool,
            )
        else:
            feature_allowed_mask = np.logical_and(
                dataset[:, idx] >= allowed_min,
                dataset[:, idx] <= allowed_max,
            )
        feature_allowed_pct = (
            100.0 * float(np.sum(feature_allowed_mask)) / float(n_rows) if n_rows else 0.0
        )
        global_allowed_mask = np.logical_and(global_allowed_mask, feature_allowed_mask)

        workflow_log(
            "   - %s (idx=%d): bounds=[%.6f, %.6f] | dataset=[%.6f, %.6f] | allowed space=%.2f%% | allowed points=%.2f%%",
            feature_label(idx),
            idx,
            allowed_min,
            allowed_max,
            dataset_min,
            dataset_max,
            coverage_pct,
            feature_allowed_pct,
        )
    global_allowed_pct = (
        100.0 * float(np.sum(global_allowed_mask)) / float(n_rows) if n_rows else 0.0
    )
    workflow_log("")
    workflow_log(
        "Global allowed samples (all numerical-feature bounds simultaneously): %.2f%% (%d/%d)",
        global_allowed_pct,
        int(np.sum(global_allowed_mask)),
        n_rows,
    )
