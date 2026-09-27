"""Pre-flight check for priority sets: do they leave a usable search space?

For every ``(dataset, priority_set)`` this reports, per sampled instance:

* whether the set builds at all (every feature declared, codes valid),
* the fraction of training rows inside the allowed region (coverage),
* the maximum attainable priority score and the sample's own score,
* whether a random point in the allowed region can reach the requested target
  within ``epsilon`` (``MINLSearchExplainer.probe_allowed_region_feasibility``),
  which separates *infeasible priorities* from *search failures* later on.

Run this before launching a study: a set with 0% feasibility will produce
empty result cells that look like method failures but are not.

Usage::

    python -m explainit.experiments.continuous_minlp.priority_sets_check
    python -m explainit.experiments.continuous_minlp.priority_sets_check \
        --datasets insurance_charges carseats --sets loose medium strict \
        --n-samples 5 --target-offset -0.3 --probe-iterations 2000
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.explainers.minlp_search import MINLSearchExplainer  # noqa: E402
from explainit.experiments.continuous_minlp.priority_sets import (  # noqa: E402
    PRIORITY_SETS,
    build_priorities,
)
from explainit.experiments.continuous_minlp.priority_methods.methods import (  # noqa: E402
    compute_priority_score,
    max_attainable_priority,
)
from explainit.experiments.continuous_minlp.priority_methods.selection import (  # noqa: E402
    load_priority_context,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUT_DIR = EXPERIMENT_DIR / "analysis"

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.priority_sets_check"
)


def _coverage(priorities: Dict[str, Any], X: np.ndarray) -> float:
    """Fraction of rows that lie inside the allowed region of every feature."""
    mask = np.ones(len(X), dtype=bool)
    for idx, cfg in priorities.get("numerical", {}).items():
        if not isinstance(cfg, dict) or cfg.get("function") is None:
            continue
        column = X[:, idx]
        mask &= (column >= float(cfg["min"])) & (column <= float(cfg["max"]))
        fn = cfg["function"]
        weights = np.array([float(np.asarray(fn(float(v))).squeeze()) for v in column])
        mask &= weights > 0.0
    for group, mapping in priorities.get("categorical", {}).items():
        allowed = {combo for combo, w in mapping.items() if w is not None and w > 0}
        if not allowed:
            return 0.0
        combos = [tuple(float(v) for v in row) for row in X[:, list(group)]]
        mask &= np.array([c in allowed for c in combos])
    return float(mask.sum()) / float(len(X)) if len(X) else 0.0


def check_set(
    pctx, set_name: str, *, sample_indices: Sequence[int], target_offset: float,
    epsilon: float, probe_iterations: int, seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    X_train = np.asarray(pctx.X_train, dtype=float)
    preds = pctx.model_predict(pctx.X_test)
    for sample_id in sample_indices:
        x = np.asarray(pctx.X_test[sample_id], dtype=float)
        row: Dict[str, Any] = {
            "dataset": pctx.dataset_key,
            "priority_set": set_name,
            "sample_id": int(sample_id),
            "prediction": float(preds[sample_id]),
            "target": float(preds[sample_id]) + float(target_offset),
        }
        try:
            priorities = build_priorities(pctx.ctx, set_name, x)
        except Exception as exc:
            row.update({"build_error": str(exc)})
            rows.append(row)
            logger.error("[%s/%s] sample %d: build failed: %s",
                         pctx.dataset_key, set_name, sample_id, exc)
            continue

        max_priority = max_attainable_priority(priorities)
        sample_priority = compute_priority_score(priorities, x)
        n_actionable = sum(
            1 for cfg in priorities["numerical"].values()
            if isinstance(cfg, dict) and cfg.get("function") is not None
        )
        n_free_groups = sum(
            1 for mapping in priorities["categorical"].values()
            if sum(1 for w in mapping.values() if w is not None) > 1
        )
        row.update({
            "build_error": None,
            "n_actionable_numerical": n_actionable,
            "n_multi_category_groups": n_free_groups,
            "max_combinations": int(2 ** n_free_groups),
            "coverage_train": _coverage(priorities, X_train),
            "max_attainable_priority": max_priority,
            "sample_priority": sample_priority,
            "sample_priority_normalised": (
                sample_priority / max_priority if max_priority else None),
        })

        try:
            explainer = MINLSearchExplainer(
                model_pred=pctx.model_predict,
                priorities=build_priorities(pctx.ctx, set_name, x),
                sample=x.tolist(),
                target=row["target"],
                dataset=X_train.copy(),
                epsilon=float(epsilon),
                feature_names=pctx.feature_names,
            )
            probe = explainer.probe_allowed_region_feasibility(
                max_iterations=int(probe_iterations), random_seed=seed)
            row.update({
                "feasible_within_epsilon": probe["feasible_within_epsilon"],
                "min_pred_distance": probe["min_pred_distance"],
                "probe_iterations": probe["iterations_run"],
                "probe_error": None,
            })
        except Exception as exc:
            row.update({
                "feasible_within_epsilon": None, "min_pred_distance": None,
                "probe_iterations": None, "probe_error": str(exc),
            })
        rows.append(row)
        logger.info(
            "[%s/%s] sample %-5d coverage=%.3f feasible=%s min_dist=%s "
            "prio=%.2f/%.2f combos<=%d",
            pctx.dataset_key, set_name, sample_id, row["coverage_train"],
            row.get("feasible_within_epsilon"),
            (f"{row['min_pred_distance']:.4f}"
             if row.get("min_pred_distance") is not None else "n/a"),
            sample_priority, max_priority or float("nan"), row["max_combinations"],
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Default: every dataset with priority sets.")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="Default: every set of each dataset.")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-offset", type=float, default=None,
                        help="Absolute offset in target units.")
    parser.add_argument("--target-offset-pred-std", type=float, default=-1.0,
                        help="Offset in multiples of the model's prediction std "
                             "(default -1.0); ignored when --target-offset is set.")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--probe-iterations", type=int, default=2000)
    parser.add_argument("--out", default=str(OUT_DIR / "priority_sets_check.csv"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("explainit").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    datasets = args.datasets or sorted(PRIORITY_SETS)
    rows: List[Dict[str, Any]] = []
    for dataset_key in datasets:
        try:
            pctx = load_priority_context(dataset_key)
        except Exception as exc:
            logger.error("[%s] context failed to load: %s", dataset_key, exc)
            continue
        rng = np.random.default_rng(args.seed)
        n_test = len(pctx.X_test)
        sample_indices = [int(i) for i in rng.permutation(n_test)[: args.n_samples]]
        if args.target_offset is not None:
            offset = float(args.target_offset)
        else:
            offset = float(args.target_offset_pred_std) * float(
                np.std(pctx.model_predict(pctx.X_test)))
        logger.info("[%s] target offset = %.4f", dataset_key, offset)
        for set_name in (args.sets or sorted(PRIORITY_SETS.get(dataset_key, {}))):
            if set_name not in PRIORITY_SETS.get(dataset_key, {}):
                continue
            rows.extend(check_set(
                pctx, set_name,
                sample_indices=sample_indices,
                target_offset=offset,
                epsilon=args.epsilon,
                probe_iterations=args.probe_iterations,
                seed=args.seed,
            ))

    _write_csv(Path(args.out), rows)
    ok = sum(1 for r in rows if r.get("feasible_within_epsilon"))
    logger.info("\n%d/%d (dataset, set, sample) cells can reach the target; "
                "wrote %s", ok, len(rows), args.out)


if __name__ == "__main__":
    main()
