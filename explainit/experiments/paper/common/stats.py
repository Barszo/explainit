"""Statistics for the paper studies.

The unit of analysis is the **paired cell**: the same dataset, priority set,
sample and target solved by two methods (or by one method under two
configurations). Method-level averages over different valid-CF subsets are not
comparable and are deliberately not offered here.

Typical use::

    df = RunRegistry("study3_comparison").load_counterfactuals()
    best = best_per_sample(df, value_col="priority_score_normalised")
    result = paired_comparison(best, "minlp", "random_search",
                               value_col="priority_score_normalised")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PAIR_KEYS = ("dataset", "priority_set", "sample_id", "seed")


@dataclass
class PairedResult:
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_delta: float
    median_delta: float
    wins_a: int
    wins_b: int
    ties: int
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]
    sign_test_p: Optional[float]
    cliffs_delta: float
    cliffs_magnitude: str
    delta_ci_low: float
    delta_ci_high: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def best_per_sample(
    df,
    *,
    value_col: str = "priority_score_normalised",
    keys: Sequence[str] = PAIR_KEYS,
    by: str = "method",
    valid_only: bool = True,
):
    """Collapse multiple CFs per sample to the best one (max ``value_col``).

    Methods are asked for different numbers of counterfactuals, so comparing
    "best of n" against "best of 1" is only fair once ``n_cfs`` is matched in
    the study config; this helper just performs the collapse.

    ``by`` is the label the comparison runs over: ``"method"`` for method
    comparisons, ``"arm"`` for ablations of one method.
    """
    frame = df[df["validity"] == True] if valid_only else df  # noqa: E712
    frame = frame[frame[value_col].notna()]
    group = [k for k in keys if k in frame.columns] + [by]
    return (
        frame.sort_values(value_col, ascending=False)
        .groupby(group, as_index=False, dropna=False)
        .first()
    )


def paired_frame(
    df, label_a: str, label_b: str, *,
    value_col: str = "priority_score_normalised",
    keys: Sequence[str] = PAIR_KEYS,
    by: str = "method",
):
    """Inner-join the two labels on ``keys`` -> one row per comparable pair."""
    keys = [k for k in keys if k in df.columns]
    a = df[df[by] == label_a][keys + [value_col]].rename(
        columns={value_col: "value_a"})
    b = df[df[by] == label_b][keys + [value_col]].rename(
        columns={value_col: "value_b"})
    merged = a.merge(b, on=keys, how="inner")
    merged["delta"] = merged["value_a"] - merged["value_b"]
    return merged


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in ``[-1, 1]`` (P(a>b) - P(a<b))."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((greater - less) / (a.size * b.size))


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if math.isnan(d):
        return "undefined"
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05,
    seed: int = 0, statistic=np.mean,
) -> tuple:
    """Percentile bootstrap CI of ``statistic`` over ``values``."""
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                        dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True)
    stats = statistic(draws, axis=1)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_comparison(
    df, label_a: str, label_b: str, *,
    value_col: str = "priority_score_normalised",
    keys: Sequence[str] = PAIR_KEYS,
    by: str = "method",
    tie_tol: float = 1e-9,
    seed: int = 0,
) -> PairedResult:
    """Full paired comparison of ``label_a`` against ``label_b``."""
    from scipy import stats as sps

    merged = paired_frame(df, label_a, label_b, value_col=value_col, keys=keys, by=by)
    a = merged["value_a"].to_numpy(dtype=float)
    b = merged["value_b"].to_numpy(dtype=float)
    delta = a - b
    wins_a = int(np.sum(delta > tie_tol))
    wins_b = int(np.sum(delta < -tie_tol))
    ties = int(delta.size - wins_a - wins_b)

    wil_stat = wil_p = None
    if delta.size >= 3 and np.any(np.abs(delta) > tie_tol):
        try:
            res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            wil_stat, wil_p = float(res.statistic), float(res.pvalue)
        except ValueError:
            pass

    sign_p = None
    if wins_a + wins_b > 0:
        sign_p = float(
            sps.binomtest(wins_a, wins_a + wins_b, 0.5, alternative="two-sided").pvalue)

    lo, hi = bootstrap_ci(delta, seed=seed)
    effect = cliffs_delta(a, b)
    return PairedResult(
        n_pairs=int(delta.size),
        mean_a=float(np.mean(a)) if a.size else float("nan"),
        mean_b=float(np.mean(b)) if b.size else float("nan"),
        mean_delta=float(np.mean(delta)) if delta.size else float("nan"),
        median_delta=float(np.median(delta)) if delta.size else float("nan"),
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        wilcoxon_stat=wil_stat,
        wilcoxon_p=wil_p,
        sign_test_p=sign_p,
        cliffs_delta=effect,
        cliffs_magnitude=cliffs_magnitude(effect),
        delta_ci_low=lo,
        delta_ci_high=hi,
    )


def holm_correction(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni adjusted p-values, preserving input order and ``None``s."""
    indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    m = len(indexed)
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, (i, p) in enumerate(sorted(indexed, key=lambda t: t[1])):
        value = min(1.0, (m - rank) * float(p))
        running = max(running, value)
        adjusted[i] = running
    return adjusted


def compare_many(
    df, label_a: str, label_b: str, *,
    group_cols: Sequence[str] = ("dataset", "priority_set"),
    value_col: str = "priority_score_normalised",
    keys: Sequence[str] = PAIR_KEYS,
    by: str = "method",
    seed: int = 0,
):
    """One paired comparison per group, with Holm-corrected p-values."""
    import pandas as pd

    group_cols = [c for c in group_cols if c in df.columns]
    rows: List[Dict[str, Any]] = []
    for group_values, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        result = paired_comparison(
            sub, label_a, label_b, value_col=value_col, keys=keys, by=by, seed=seed)
        row = dict(zip(group_cols, group_values))
        row.update(result.as_dict())
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["wilcoxon_p_holm"] = holm_correction(list(frame["wilcoxon_p"]))
        frame["sign_test_p_holm"] = holm_correction(list(frame["sign_test_p"]))
    return frame


__all__ = [
    "PAIR_KEYS",
    "PairedResult",
    "best_per_sample",
    "bootstrap_ci",
    "cliffs_delta",
    "cliffs_magnitude",
    "compare_many",
    "holm_correction",
    "paired_comparison",
    "paired_frame",
]
