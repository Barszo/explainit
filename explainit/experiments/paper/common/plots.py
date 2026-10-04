"""Figures for the paper studies.

Deliberately small: three plot types cover what studies I-III need. Every
function takes a tidy DataFrame (as produced by
``RunRegistry.load_counterfactuals``) and returns the Matplotlib axes, so
notebooks can restyle or save them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def sensitivity_curve(
    df,
    *,
    x_col: str,
    value_col: str = "priority_score_normalised",
    hue_col: Optional[str] = "priority_set",
    valid_only: bool = True,
    ax=None,
):
    """Mean +/- s.e. of ``value_col`` against a swept hyperparameter."""
    frame = df[df["validity"] == True] if valid_only else df  # noqa: E712
    frame = frame[frame[value_col].notna()]
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    groups = [(None, frame)] if hue_col is None else list(frame.groupby(hue_col))
    for label, sub in groups:
        stats = sub.groupby(x_col)[value_col].agg(["mean", "sem", "count"]).reset_index()
        ax.errorbar(stats[x_col], stats["mean"], yerr=stats["sem"].fillna(0.0),
                    marker="o", capsize=3, label=str(label))
    ax.set_xlabel(x_col)
    ax.set_ylabel(value_col)
    if hue_col is not None:
        ax.legend(title=hue_col)
    ax.grid(alpha=0.3)
    return ax


def budget_pareto(
    df,
    *,
    cost_col: str = "model_rows",
    value_col: str = "priority_score_normalised",
    label_col: str = "cell",
    hue_col: Optional[str] = "method",
    valid_only: bool = True,
    ax=None,
):
    """Priority achieved against the model-evaluation budget spent."""
    frame = df[df["validity"] == True] if valid_only else df  # noqa: E712
    frame = frame[frame[value_col].notna()]
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    group_cols = [c for c in [hue_col, label_col] if c is not None]
    stats = frame.groupby(group_cols)[[cost_col, value_col]].mean().reset_index()
    hues = [(None, stats)] if hue_col is None else list(stats.groupby(hue_col))
    for label, sub in hues:
        sub = sub.sort_values(cost_col)
        ax.plot(sub[cost_col], sub[value_col], marker="o", label=str(label))
        for _, row in sub.iterrows():
            ax.annotate(str(row[label_col]), (row[cost_col], row[value_col]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(f"{cost_col} (mean per sample, log scale)")
    ax.set_ylabel(value_col)
    if hue_col is not None:
        ax.legend(title=hue_col)
    ax.grid(alpha=0.3)
    return ax


def paired_delta_plot(
    merged,
    *,
    delta_col: str = "delta",
    label: str = "MINLP - baseline",
    ax=None,
):
    """Sorted per-sample paired deltas (the figure that carries the claim)."""
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    values = np.sort(np.asarray(merged[delta_col], dtype=float))
    colours = ["tab:green" if v > 0 else "tab:red" for v in values]
    ax.bar(np.arange(values.size), values, color=colours)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired cell (sorted)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3, axis="y")
    return ax


__all__ = ["budget_pareto", "paired_delta_plot", "sensitivity_curve"]
