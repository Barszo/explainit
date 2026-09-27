"""Summarise a finished study: per-arm table + paired comparisons.

Usage::

    # ablation: every arm against one reference arm
    python -m explainit.experiments.paper.analyse_study --study step2_validation \
        --by arm --reference baseline_no_step_control

    # method comparison: every method against a baseline method
    python -m explainit.experiments.paper.analyse_study --study study3_comparison \
        --by method --reference random_search

Writes ``analysis/summary.csv`` and ``analysis/paired.csv`` next to the study's
``index.csv`` and prints both tables.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.paper.common import stats  # noqa: E402
from explainit.experiments.paper.common.registry import (  # noqa: E402
    RunRegistry,
    current_run_ids,
)

logger = logging.getLogger("explainit.experiments.paper.analyse_study")


def summarise(df, by: str, value_col: str):
    import pandas as pd

    produced = df[df["cf_prediction"].notna()]
    valid = df[df["validity"] == True]  # noqa: E712
    rows = []
    for label, sub in df.groupby(by, dropna=False):
        sub_valid = valid[valid[by] == label]
        sub_produced = produced[produced[by] == label]
        n_samples = sub[["dataset", "priority_set", "sample_id", "seed"]].drop_duplicates().shape[0]
        row = {
            by: label,
            "n_samples": n_samples,
            "n_cfs_produced": len(sub_produced),
            "n_cfs_valid": len(sub_valid),
            "validity_rate": (len(sub_valid) / len(sub_produced)) if len(sub_produced) else 0.0,
            f"mean_{value_col}": sub_valid[value_col].mean() if len(sub_valid) else float("nan"),
            "mean_priority_score": sub_valid["priority_score"].mean() if len(sub_valid) else float("nan"),
            "mean_abs_pred_error": sub_valid["abs_pred_error"].mean() if len(sub_valid) else float("nan"),
            "mean_l1": sub_valid["l1"].mean() if len(sub_valid) else float("nan"),
            "mean_model_rows": sub["model_rows"].mean(),
            "mean_time_seconds": sub["time_seconds"].mean(),
        }
        if "cf_source" in sub.columns:
            sources = sub["cf_source"].dropna()
            anchor = sources.astype(str).str.startswith("anchor").sum()
            row["anchor_fraction"] = (anchor / len(sources)) if len(sources) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"mean_{value_col}", ascending=False)


def paired_table(df, by: str, reference: str, value_col: str, seed: int):
    import pandas as pd

    labels = [l for l in df[by].dropna().unique() if l != reference]
    rows = []
    for label in sorted(labels):
        result = stats.paired_comparison(
            df, label, reference, value_col=value_col, by=by, seed=seed)
        row = {by: label, "reference": reference}
        row.update(result.as_dict())
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["wilcoxon_p_holm"] = stats.holm_correction(list(frame["wilcoxon_p"]))
        frame["sign_test_p_holm"] = stats.holm_correction(list(frame["sign_test_p"]))
    return frame


def diagnostics(df, by: str):
    """Where the method's output actually comes from, and how it fails.

    Columns answer, per arm/method: how often no counterfactual was produced,
    how often the returned one is the Stage 1 anchor rather than the
    optimiser's output, which exemplar strategy was used, why the loop
    stopped, and how wrong the Shapley surrogate was at the warm start
    (``warm_start_best_model_gap`` against ``epsilon``).
    """
    import pandas as pd

    rows = []
    for label, sub in df.groupby(by, dropna=False):
        n = len(sub)
        produced = sub[sub["cf_prediction"].notna()]
        sources = sub["cf_source"].dropna().astype(str) if "cf_source" in sub else pd.Series(dtype=str)
        exemplars = sub["exemplar_source"].dropna().astype(str) if "exemplar_source" in sub else pd.Series(dtype=str)
        stops = sub["stop_reason"].dropna().astype(str) if "stop_reason" in sub else pd.Series(dtype=str)
        gaps = pd.to_numeric(sub.get("warm_start_best_model_gap"), errors="coerce").dropna()
        eps = pd.to_numeric(sub.get("epsilon"), errors="coerce").dropna()
        eps_value = float(eps.iloc[0]) if len(eps) else float("nan")
        row = {
            by: label,
            "n_rows": n,
            "no_cf_rate": 1.0 - (len(produced) / n if n else 0.0),
            "invalid_cf_rate": (
                1.0 - (produced["validity"].sum() / len(produced)) if len(produced) else float("nan")),
            "anchor_rate": (
                sources.str.startswith("anchor").mean() if len(sources) else float("nan")),
            "peak_lock_in_rate": (
                sources.str.contains("peak_lock_in").mean() if len(sources) else float("nan")),
            "exemplar_random_rate": (
                (exemplars == "random_in_range").mean() if len(exemplars) else float("nan")),
            "exemplar_dataset_rate": (
                (exemplars == "dataset_priority_filtered").mean() if len(exemplars) else float("nan")),
            "stop_search_failed_rate": (
                (stops == "search_failed").mean() if len(stops) else float("nan")),
            "median_warm_start_gap": gaps.median() if len(gaps) else float("nan"),
            "warm_start_gap_over_eps_rate": (
                (gaps > eps_value).mean() if len(gaps) and np.isfinite(eps_value) else float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("anchor_rate", ascending=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True)
    parser.add_argument("--by", default="arm", choices=["arm", "method", "cell"])
    parser.add_argument("--reference", default=None,
                        help="Label every other one is compared against.")
    parser.add_argument("--value", default="priority_score_normalised")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--include-stale", action="store_true",
                        help="Also read runs that the current config no longer defines.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import pandas as pd

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    root = Path(args.results_root) / args.study if args.results_root else None
    registry = RunRegistry(args.study, root=root)
    df = registry.load_counterfactuals(
        run_ids=None if args.include_stale else current_run_ids(args.study))
    if df.empty:
        raise SystemExit(f"No results found for study '{args.study}' in {registry.root}")

    pd.set_option("display.width", 200)
    summary = summarise(df, args.by, args.value)
    print(f"\n=== per-{args.by} summary ({args.study}) ===")
    print(summary.round(4).to_string(index=False))

    out_dir = registry.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    diag = diagnostics(df, args.by)
    print(f"\n=== failure / provenance diagnostics ({args.study}) ===")
    print(diag.round(4).to_string(index=False))
    diag.to_csv(out_dir / "diagnostics.csv", index=False)

    if args.reference:
        best = stats.best_per_sample(df, value_col=args.value, by=args.by)
        paired = paired_table(best, args.by, args.reference, args.value, args.seed)
        print(f"\n=== paired vs '{args.reference}' (value={args.value}) ===")
        cols = [args.by, "n_pairs", "mean_a", "mean_b", "mean_delta", "wins_a",
                "wins_b", "ties", "wilcoxon_p", "wilcoxon_p_holm", "sign_test_p",
                "cliffs_delta", "cliffs_magnitude", "delta_ci_low", "delta_ci_high"]
        print(paired[[c for c in cols if c in paired.columns]].round(4).to_string(index=False))
        paired.to_csv(out_dir / "paired.csv", index=False)
    logger.info("\nWrote %s", out_dir)


if __name__ == "__main__":
    main()
