"""Priority plot of one categorical (one-hot) feature group.

Without data it is a bar chart of the priority weights, marking the
sample's and exemplar's category. With a dataset's features and model
targets it shows the targets per category instead: stacked class counts for a
categorical target, box plots for a continuous one.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from explainit.plotting._common import (
    PathLike,
    bar_width,
    draw_bar_label,
    draw_bars,
    draw_connector,
    ensure_dir,
    finalize,
    group_path,
    group_title,
    infer_target_type,
    normalize_vectors,
    pair_count,
    pair_label,
)
from explainit.plotting.style import (
    TEXT,
    apply_style,
    bar_color,
    pair_color,
    style_axes,
    styled_legend,
)


def plot_categorical_priority(
    group_indices: tuple[int, ...],
    mapping: Mapping[tuple[Any, ...], float],
    *,
    sample: Any | None = None,
    exemplar: Any | None = None,
    feature_matrix: Sequence[Sequence[float]] | None = None,
    target_values: Sequence[float] | None = None,
    target_name: str = "target",
    feature_names: Sequence[str] | None = None,
    save_path: PathLike | None = None,
    save_dir: PathLike | None = None,
    show: bool = False,
) -> plt.Figure | None:
    """Plot the priority weights of the categorical group ``group_indices``.

    Args:
        group_indices: Column indices of the one-hot group.
        mapping: ``{combination: weight}``; ``None`` weights are shown as 0.
        sample, exemplar: One feature vector each, or several (2-D); their
            categories are highlighted, each pair in its own colour.
        feature_matrix, target_values: Dataset rows and model targets; when
            both are given, the targets per category are shown instead.
        target_name: Name of the target in labels.
        feature_names: Names of all columns, used in the title.
        save_path: Write the figure to this file.
        save_dir: Write the figure to a default file name in this directory.
        show: Show the figure; otherwise close it.

    Returns:
        The figure, or ``None`` if ``mapping`` is empty.
    """
    if not mapping:
        return None

    apply_style()
    categories = list(mapping.keys())
    weights = np.array([0.0 if v is None else float(v) for v in mapping.values()], dtype=float)
    labels = [str(cat) for cat in categories]

    data = _dataset(group_indices, feature_matrix, target_values)
    if data is not None:
        fig, has_legend = _target_figure(categories, weights, labels, *data, target_name)
    else:
        samples = normalize_vectors(sample)
        exemplars = normalize_vectors(exemplar)
        fig, has_legend = _weights_figure(
            group_indices, categories, weights, labels, samples, exemplars, feature_names
        )
    if has_legend:
        fig.subplots_adjust(right=0.75)

    if save_path is None and save_dir is not None:
        save_path = group_path(ensure_dir(save_dir), "priority_categorical", group_indices)
    finalize(fig, save_path, show)
    return fig


def _dataset(group_indices, feature_matrix, target_values):
    """``(combinations, targets, target_type)`` for rows with a finite target, or ``None``."""
    if feature_matrix is None or target_values is None:
        return None
    rows = np.asarray(feature_matrix, dtype=float)
    targets = np.asarray(target_values, dtype=float).flatten()
    if not (rows.ndim == 2 and len(group_indices) > 0 and targets.size == rows.shape[0]):
        return None
    combinations = [tuple(row[list(group_indices)]) for row in rows]
    mask = np.isfinite(targets)
    kept = [c for c, keep in zip(combinations, mask, strict=False) if keep]
    return kept, targets[mask], infer_target_type(targets)


def _target_figure(categories, weights, labels, combinations, targets, target_type, target_name):
    """Targets per category; returns ``(figure, has_legend)``."""
    fig, ax = plt.subplots(figsize=(12, 8))
    num_categories = len(categories)
    in_category = [np.array([c == cat for c in combinations], dtype=bool) for cat in categories]

    if target_type == "categorical":
        bottoms = np.zeros(num_categories, dtype=float)
        for j, cls in enumerate(np.unique(targets)):
            counts = np.array(
                [float(np.sum(mask & (targets == cls))) for mask in in_category], dtype=float
            )
            ax.bar(
                np.arange(num_categories),
                counts,
                bottom=bottoms,
                width=bar_width(num_categories),
                color=bar_color(j),
                edgecolor=TEXT,
                linewidth=1.2,
                alpha=0.8,
                label=f"{target_name}={cls}",
            )
            bottoms += counts
        for i, total in enumerate(bottoms):
            ax.text(
                i,
                total + max(bottoms) * 0.02 if max(bottoms) > 0 else 0.1,
                f"p={weights[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color=TEXT,
            )
        ax.set_ylabel("count")
        ax.set_title(f"{target_name} counts by feature group")
    else:
        grouped = []
        tick_labels = []
        for i, mask in enumerate(in_category):
            vals = targets[mask]
            grouped.append(vals if vals.size else np.array([np.nan]))
            tick_labels.append(f"{labels[i]}\np={weights[i]:.3f}")
        boxes = ax.boxplot(grouped, patch_artist=True, tick_labels=tick_labels)
        for i, box in enumerate(boxes["boxes"]):
            box.set_facecolor(bar_color(i))
            box.set_edgecolor(TEXT)
            box.set_alpha(0.75)
        for median in boxes["medians"]:
            median.set_color(TEXT)
            median.set_linewidth(2.0)
        ax.set_ylabel(target_name)
        ax.set_title(f"{target_name} distribution by feature group")
    ax.set_xlabel("Feature group values")
    if target_type == "categorical":
        ax.set_xticks(range(num_categories))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        styled_legend(ax, fontsize=12, anchor=(1.02, 0.9))
    style_axes(ax, grid="y")
    fig.tight_layout()
    return fig, True


def _category_index(group_indices, categories, vector):
    """Index of the category ``vector`` has in the group, or ``None``."""
    if vector is None:
        return None
    try:
        combination = tuple(float(vector[gi]) for gi in group_indices)
    except (IndexError, TypeError, ValueError):
        return None
    for i, category in enumerate(categories):
        try:
            if all(
                math.isclose(float(a), float(b), abs_tol=1e-9)
                for a, b in zip(combination, category, strict=False)
            ):
                return i
        except (TypeError, ValueError):
            if combination == category:
                return i
    return None


def _weights_figure(group_indices, categories, weights, labels, samples, exemplars, names):
    """Bar chart of the weights; returns ``(figure, has_legend)``."""
    fig, ax = plt.subplots(figsize=(12, 8))
    num_categories = len(categories)
    bars = draw_bars(ax, weights)

    weights_max = float(max(weights)) if num_categories else 0.0
    n_pairs = pair_count(samples, exemplars)
    sample_indices = []
    for pi in range(n_pairs):
        color = pair_color(pi)
        s_vec = samples[pi] if samples is not None and pi < len(samples) else None
        e_vec = exemplars[pi] if exemplars is not None and pi < len(exemplars) else None
        s_idx = _category_index(group_indices, categories, s_vec)
        e_idx = _category_index(group_indices, categories, e_vec)
        sample_indices.append(s_idx)
        if s_idx is not None and e_idx is not None and s_idx != e_idx:
            y_top = max(weights[s_idx], weights[e_idx]) + (weights_max * 0.18 + 0.05)
            draw_connector(ax, [s_idx, e_idx], [y_top, y_top], color)
        for index, marker, name, lift in (
            (s_idx, "v", "Sample", weights_max * 0.05 + 0.02),
            (e_idx, "s", "Exemplar", weights_max * 0.12 + 0.04),
        ):
            if index is None:
                continue
            bars[index].set_edgecolor(color)
            bars[index].set_linewidth(4)
            ax.plot(
                index,
                weights[index] + lift,
                marker,
                color=color,
                markersize=15,
                markeredgecolor=TEXT,
                markeredgewidth=2,
                label=pair_label(name, pi, n_pairs),
            )

    if num_categories == 1:
        ax.set_xlim(-1, 1)

    current = {i for i in sample_indices if i is not None}
    offset = weights_max * 0.01 if weights_max else 0.01
    for i, (bar, weight) in enumerate(zip(bars, weights, strict=False)):
        text = f"{weight:.3f}" + (" (Current)" if i in current else "")
        draw_bar_label(ax, bar, text, offset)

    ax.set_xlabel("Category Combinations")
    ax.set_ylabel("Priority Weight")
    ax.set_title(group_title("Priority Weights", group_indices, names))
    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        styled_legend(ax, handles, legend_labels, fontsize=12, anchor=(1.02, 0.9))
    style_axes(ax, grid="y")
    fig.tight_layout()
    return fig, bool(handles)
