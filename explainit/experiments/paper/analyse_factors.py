"""Badania II: paired effect of each factor level against its baseline.

Every variant dataset (``<base>__<factor>_<level>``) is compared against its
own base dataset on the *same* test instances and seeds, separately for each
method. Three deltas are reported because a factor can hurt in three different
ways:

* ``priority``  -- the counterfactual is less preferable,
* ``validity``  -- the method more often fails to reach the target at all,
* ``model_rows``-- the same answer costs more model evaluations.

Statistics: Wilcoxon signed-rank and sign test on the paired differences,
Cliff's delta as effect size, a percentile bootstrap CI of the mean difference,
and Holm correction across the levels of one factor.

Usage::

    python -m explainit.experiments.paper.analyse_factors --study study2_factors
    python -m explainit.experiments.paper.analyse_factors --study study2_factors \
        --method minlp --value priority_score_normalised
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.paper.common import stats  # noqa: E402
from explainit.experiments.paper.common.registry import (  # noqa: E402
    RunRegistry,
    current_run_ids,
)

logger = logging.getLogger("explainit.experiments.paper.analyse_factors")

PAIR_ON = ("sample_id", "seed", "priority_set")


def split_variant(dataset: str):
    """``('auto_mpg__train_size_0p5')`` -> ``('auto_mpg', 'train_size', '0p5')``."""
    if "__" not in dataset:
        return dataset, "baseline", "base"
    base, rest = dataset.split("__", 1)
    factor, _, level = rest.rpartition("_")
    return base, factor, level


def annotate(df):
    parts = df["dataset"].map(split_variant)
    df = df.copy()
    df["base_dataset"] = [p[0] for p in parts]
    df["factor"] = [p[1] for p in parts]
    df["level"] = [p[2] for p in parts]
    return df


def _paired(sub_variant, sub_base, value_col: str, seed: int) -> Dict[str, Any]:
    keys = [k for k in PAIR_ON if k in sub_variant.columns]
    a = sub_variant[keys + [value_col]].rename(columns={value_col: "value_a"})
    b = sub_base[keys + [value_col]].rename(columns={value_col: "value_b"})
    merged = a.merge(b, on=keys, how="inner").dropna(subset=["value_a", "value_b"])
    if merged.empty:
        return {"n_pairs": 0}
    from scipy import stats as sps

    va = merged["value_a"].to_numpy(dtype=float)
    vb = merged["value_b"].to_numpy(dtype=float)
    delta = va - vb
    wins = int(np.sum(delta > 1e-9))
    losses = int(np.sum(delta < -1e-9))
    wil_p = None
    if delta.size >= 3 and np.any(np.abs(delta) > 1e-9):
        try:
            wil_p = float(sps.wilcoxon(va, vb).pvalue)
        except ValueError:
            wil_p = None
    sign_p = (float(sps.binomtest(wins, wins + losses, 0.5).pvalue)
              if wins + losses else None)
    lo, hi = stats.bootstrap_ci(delta, seed=seed)
    return {
        "n_pairs": int(delta.size),
        "variant_mean": float(np.mean(va)),
        "base_mean": float(np.mean(vb)),
        "delta": float(np.mean(delta)),
        "worse": wins if False else losses,  # kept explicit: losses = variant worse
        "better": wins,
        "wilcoxon_p": wil_p,
        "sign_test_p": sign_p,
        "cliffs_delta": stats.cliffs_delta(va, vb),
        "ci_low": lo,
        "ci_high": hi,
    }


def factor_table(df, *, value_col: str, method: Optional[str], seed: int):
    import pandas as pd

    df = annotate(df)
    if method:
        df = df[df["method"] == method]
    # Validity as a numeric column so the same pairing works for it.
    df["validity_num"] = df["validity"].astype(float)

    rows: List[Dict[str, Any]] = []
    for base_dataset, group in df.groupby("base_dataset"):
        base = group[group["factor"] == "baseline"]
        if base.empty:
            continue
        for (factor, level), variant in group[group["factor"] != "baseline"].groupby(
                ["factor", "level"]):
            row = {
                "base_dataset": base_dataset,
                "factor": factor,
                "level": level,
                "method": method or "all",
            }
            row.update({f"prio_{k}": v for k, v in
                        _paired(variant, base, value_col, seed).items()})
            row.update({f"valid_{k}": v for k, v in
                        _paired(variant, base, "validity_num", seed).items()})
            row.update({f"rows_{k}": v for k, v in
                        _paired(variant, base, "model_rows", seed).items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Holm correction within each (base_dataset, factor) family.
    frame["prio_wilcoxon_p_holm"] = np.nan
    for (_bd, _f), idx in frame.groupby(["base_dataset", "factor"]).groups.items():
        sub = frame.loc[idx]
        frame.loc[idx, "prio_wilcoxon_p_holm"] = stats.holm_correction(
            list(sub["prio_wilcoxon_p"]))
    return frame.sort_values(["base_dataset", "factor", "level"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="study2_factors")
    parser.add_argument("--value", default="priority_score_normalised")
    parser.add_argument("--method", nargs="*", default=["minlp", "random_search"])
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--include-stale", action="store_true",
                        help="Also read runs that the current config no longer defines.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import pandas as pd

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    pd.set_option("display.width", 220)

    root = Path(args.results_root) / args.study if args.results_root else None
    registry = RunRegistry(args.study, root=root)
    df = registry.load_counterfactuals(
        run_ids=None if args.include_stale else current_run_ids(args.study))
    if df.empty:
        raise SystemExit(f"No results for study '{args.study}'.")

    out_dir = registry.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for method in args.method:
        table = factor_table(df, value_col=args.value, method=method, seed=args.seed)
        if table.empty:
            continue
        frames.append(table)
        cols = ["base_dataset", "factor", "level", "prio_n_pairs", "prio_base_mean",
                "prio_variant_mean", "prio_delta", "prio_better", "prio_worse",
                "prio_wilcoxon_p", "prio_wilcoxon_p_holm", "prio_cliffs_delta",
                "valid_base_mean", "valid_variant_mean", "valid_delta",
                "rows_delta"]
        print(f"\n=== factor effects for method '{method}' (value={args.value}) ===")
        print(table[[c for c in cols if c in table.columns]].round(4).to_string(index=False))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out_dir / "factor_effects.csv", index=False)
        logger.info("\nWrote %s", out_dir / "factor_effects.csv")


if __name__ == "__main__":
    main()
