"""Priority plot of one numerical feature.

Without data, the plot shows the priority curve. With a dataset's feature
values and model targets it adds context, depending on the data:

* continuous target, continuous feature: 2-D density of feature vs. target
  above the priority curve;
* continuous target, discrete feature: one target histogram per feature value
  above a bar chart of the priorities;
* categorical target: class-conditional feature histograms with the priority
  curve on a second y-axis.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from explainit.plotting._common import (
    PathLike,
    PriorityCurve,
    draw_connector,
    draw_curve,
    draw_level_lines,
    draw_pairs_on_curve,
    ensure_dir,
    finalize,
    infer_target_type,
    is_actionable_numeric,
    normalize_target_values,
    normalize_vectors,
    numerical_path,
    pair_count,
    pair_label,
    values_at,
)
from explainit.plotting.style import (
    TEXT,
    apply_style,
    bar_color,
    line_color,
    pair_color,
    style_axes,
    styled_legend,
)


def plot_numerical_priority(
    idx: int,
    constraint: Mapping[str, Any],
    *,
    sample: Any | None = None,
    exemplar: Any | None = None,
    sample_target_value: Any | None = None,
    exemplar_target_value: Any | None = None,
    feature_values: Sequence[float] | None = None,
    target_values: Sequence[float] | None = None,
    target_name: str = "target",
    target_threshold: float | None = None,
    feature_name: str | None = None,
    save_path: PathLike | None = None,
    save_dir: PathLike | None = None,
    show: bool = False,
) -> plt.Figure | None:
    """Plot the priority of numerical feature ``idx``.

    Args:
        idx: Column index of the feature.
        constraint: Its priority (``{"min", "max", "function"}``).
        sample, exemplar: One feature vector each, or several (2-D) to mark
            several sample/exemplar pairs, each pair in its own colour.
        sample_target_value, exemplar_target_value: Model output of the
            sample(s)/exemplar(s), used to place them on the density plot.
        feature_values, target_values: Dataset column and model targets; when
            both are given the plot adds the data context (module docstring).
        target_name: Name of the target in labels.
        target_threshold: Decision threshold drawn on the target axis.
        feature_name: Name of the feature in titles and file names.
        save_path: Write the figure to this file.
        save_dir: Write the figure to a default file name in this directory.
        show: Show the figure; otherwise close it.

    Returns:
        The figure, or ``None`` if the feature is immutable.
    """
    if not is_actionable_numeric(constraint):
        return None

    apply_style()
    curve = PriorityCurve.from_constraint(constraint, points=1200)
    title_name = feature_name if feature_name else f"Feature {idx}"

    samples_list = normalize_vectors(sample)
    exemplars_list = normalize_vectors(exemplar)
    n_pairs = pair_count(samples_list, exemplars_list)
    samples = values_at(samples_list, n_pairs, idx)
    exemplars = values_at(exemplars_list, n_pairs, idx)
    data = _dataset(feature_values, target_values)

    if data is None:
        fig = _curve_figure(curve, title_name, samples, exemplars)
    else:
        features, targets = data
        if infer_target_type(targets) == "continuous":
            plt.subplots(figsize=(12, 8))  # left open: known issue PLOT-2
            if np.unique(features).size <= 12:
                fig = _discrete_feature_figure(
                    curve, title_name, samples, exemplars, features, targets,
                    target_name, target_threshold,
                )  # fmt: skip
            else:
                fig = _density_figure(
                    curve, title_name, samples, exemplars,
                    normalize_target_values(sample_target_value, n_pairs),
                    normalize_target_values(exemplar_target_value, n_pairs),
                    features, targets, target_name, target_threshold,
                )  # fmt: skip
        else:
            fig = _class_target_figure(
                curve, title_name, samples, exemplars, features, targets, target_name
            )

    if save_path is None and save_dir is not None:
        save_path = numerical_path(ensure_dir(save_dir), "priority_function", idx, feature_name)
    finalize(fig, save_path, show)
    return fig


def _dataset(feature_values, target_values):
    """Finite ``(features, targets)`` pairs, or ``None`` if fewer than three."""
    if feature_values is None or target_values is None:
        return None
    features = np.asarray(feature_values, dtype=float).flatten()
    targets = np.asarray(target_values, dtype=float).flatten()
    mask = np.isfinite(features) & np.isfinite(targets)
    features, targets = features[mask], targets[mask]
    if features.size < 3 or targets.size != features.size:
        return None
    return features, targets


def _curve_figure(curve, title_name, samples, exemplars):
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_curve(ax, curve)
    draw_pairs_on_curve(ax, curve, samples, exemplars, size=12, edge_width=2)
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Priority Weight")
    ax.set_title(f"Priority Function for Numerical {title_name}")
    styled_legend(ax, fontsize=16, anchor=(1.02, 0.9))
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    return fig


def _discrete_feature_figure(
    curve, title_name, samples, exemplars, features, targets, target_name, target_threshold
):
    """One target histogram per feature value, above a bar chart of their priorities."""
    f = curve.function
    ordered_vals = np.sort(np.unique(features))
    grouped = [targets[np.isclose(features, v, atol=1e-9)] for v in ordered_vals]
    n_groups = len(ordered_vals)
    fig, axes = plt.subplots(
        n_groups + 1, 1, figsize=(16, max(28, 10.4 * (n_groups + 1))), sharex=False
    )
    hist_axes, ax_bottom = axes[:-1], axes[-1]

    all_vals = np.concatenate([g for g in grouped if g.size > 0]) if grouped else np.array([])
    if all_vals.size > 0:
        bins = min(50, max(10, int(np.sqrt(all_vals.size))))
        x_min, x_max = float(np.min(all_vals)), float(np.max(all_vals))
    else:
        bins = 20
        x_min, x_max = 0.0, 1.0

    # A common density scale for all histograms.
    y_max = 1.0
    for vals in grouped:
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=bins, range=(x_min, x_max), density=True)
        if hist.size:
            y_max = max(y_max, float(np.max(hist)))

    for i, (v, vals) in enumerate(zip(ordered_vals, grouped, strict=False)):
        ax_i = hist_axes[i]
        if vals.size == 0:
            ax_i.text(
                0.5, 0.5, "No samples", transform=ax_i.transAxes, ha="center", va="center",
                color=TEXT,
            )  # fmt: skip
        else:
            ax_i.hist(
                vals,
                bins=bins,
                range=(x_min, x_max),
                density=True,
                alpha=0.55,
                color=bar_color(i),
                edgecolor=TEXT,
                linewidth=1.0,
            )
        if target_threshold is not None and np.isfinite(target_threshold):
            ax_i.axvline(
                float(target_threshold),
                color=TEXT,
                linewidth=1.8,
                linestyle="-.",
                label=f"{target_name} threshold ({float(target_threshold):.3f})",
            )
        pv = float(np.asarray(f(v)).squeeze())
        ax_i.set_xlim(x_min, x_max)
        ax_i.set_ylim(0.0, y_max * 1.05)
        ax_i.set_ylabel("density")
        ax_i.set_title(f"{title_name}={v:.3f}  |  p={pv:.2f}", fontsize=24)
        if i == n_groups - 1:
            ax_i.set_xlabel(target_name, labelpad=8)
        style_axes(ax_i)
        ax_i.tick_params(axis="both", labelsize=24)

    handles, _ = hist_axes[0].get_legend_handles_labels()
    if handles:
        styled_legend(hist_axes[0], fontsize=24, anchor=(1.10, 0.85))

    pvals = np.array([float(np.asarray(f(v)).squeeze()) for v in ordered_vals], dtype=float)
    bars = ax_bottom.bar(
        range(len(ordered_vals)),
        pvals,
        color=[bar_color(i) for i in range(len(ordered_vals))],
        edgecolor=TEXT,
        linewidth=1.8,
        alpha=0.85,
    )
    for i, b in enumerate(bars):
        ax_bottom.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"p={pvals[i]:.2f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=26,
        )
    _draw_pairs_on_bars(ax_bottom, ordered_vals, pvals, samples, exemplars)
    ax_bottom.set_xticks(range(len(ordered_vals)))
    ax_bottom.set_xticklabels([f"{v:.3f}" for v in ordered_vals], rotation=30, ha="right")
    ax_bottom.set_ylim(-0.02, 1.02)
    ax_bottom.set_xlabel("Feature Value")
    ax_bottom.set_ylabel("Priority Weight")
    ax_bottom.set_title(f"Priority by observed {title_name} values", fontsize=26)
    ax_bottom.tick_params(axis="both", labelsize=24)
    handles, _ = ax_bottom.get_legend_handles_labels()
    if handles:
        styled_legend(ax_bottom, fontsize=24, anchor=(1.10, 0.82))
    style_axes(ax_bottom)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80, hspace=0.38)
    return fig


def _draw_pairs_on_bars(ax, ordered_vals, pvals, samples, exemplars):
    """Mark samples/exemplars on the bar of the feature value they have."""

    def bar_index(value):
        if value is None:
            return None
        match = np.where(np.isclose(ordered_vals, value, atol=1e-9))[0]
        return int(match[0]) if match.size else None

    n_pairs = len(samples)
    for pi in range(n_pairs):
        color = pair_color(pi)
        s_idx, e_idx = bar_index(samples[pi]), bar_index(exemplars[pi])
        if s_idx is not None and e_idx is not None:
            draw_connector(ax, [s_idx, e_idx], [pvals[s_idx], pvals[e_idx]], color)
        for index, marker, name in ((s_idx, "v", "Sample"), (e_idx, "s", "Exemplar")):
            if index is None:
                continue
            ax.plot(
                index,
                pvals[index],
                marker,
                color=color,
                markersize=12,
                markeredgecolor=TEXT,
                markeredgewidth=1.5,
                label=pair_label(name, pi, n_pairs),
            )


def _density_figure(
    curve,
    title_name,
    samples,
    exemplars,
    sample_targets,
    exemplar_targets,
    features,
    targets,
    target_name,
    target_threshold,
):
    """2-D density of feature vs. target, above the priority curve."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [3, 2]}
    )
    bins_x = min(60, max(10, int(np.sqrt(features.size))))
    bins_y = min(60, max(10, int(np.sqrt(targets.size))))
    h = ax_top.hist2d(features, targets, bins=[bins_x, bins_y], cmap="viridis")
    cbar = fig.colorbar(h[3], ax=ax_top)
    cbar.ax.tick_params(colors=TEXT)
    cbar.set_label("density", color=TEXT)

    if not curve.is_constant:
        draw_level_lines(ax_top, curve, linewidth=2.2, alpha=0.95, label="priority")
    if target_threshold is not None and np.isfinite(target_threshold):
        ax_top.axhline(
            float(target_threshold),
            color=TEXT,
            linewidth=2.0,
            linestyle="-.",
            label=f"{target_name} threshold ({float(target_threshold):.3f})",
        )

    # Samples/exemplars without a known target sit at the median target.
    median_y = float(np.median(targets))
    n_pairs = len(samples)
    for pi in range(n_pairs):
        color = pair_color(pi)
        s_v, e_v = samples[pi], exemplars[pi]
        s_t = sample_targets[pi] if sample_targets is not None else None
        e_t = exemplar_targets[pi] if exemplar_targets is not None else None
        s_y = s_t if s_t is not None else median_y
        e_y = e_t if e_t is not None else median_y
        if s_v is not None and np.isfinite(s_v) and e_v is not None and np.isfinite(e_v):
            draw_connector(ax_top, [s_v, e_v], [s_y, e_y], color)
        for value, y, marker, name in ((s_v, s_y, "v", "sample"), (e_v, e_y, "s", "exemplar")):
            if value is None or not np.isfinite(value):
                continue
            ax_top.scatter(
                [value],
                [y],
                marker=marker,
                s=130,
                color=color,
                edgecolors=TEXT,
                linewidths=1.5,
                label=f"{pair_label(name, pi, n_pairs)} x={value:.3f}",
            )

    ax_top.set_ylabel(target_name)
    ax_top.set_xlabel("Feature Value", labelpad=10)
    ax_top.set_title(f"Priority + {title_name} vs {target_name} density")
    styled_legend(ax_top, fontsize=12, anchor=(1.24, 0.9))
    if curve.is_constant:
        ax_top.text(
            1.24,
            0.18,
            "Note: priority is constant on this range;\nlevel lines hidden.",
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            color=TEXT,
            fontsize=11,
        )
    style_axes(ax_top)
    ax_top.tick_params(axis="x", labelbottom=True)

    draw_curve(ax_bottom, curve)
    draw_pairs_on_curve(ax_bottom, curve, samples, exemplars, size=10, edge_width=1.5)
    if not curve.is_constant:
        draw_level_lines(ax_bottom, curve, linewidth=1.8, alpha=0.9, label="boundary p")
    ax_bottom.set_xlabel("Feature Value")
    ax_bottom.set_ylabel("Priority Weight")
    ax_bottom.set_title(f"Priority Function for Numerical {title_name}")
    styled_legend(ax_bottom, fontsize=11, anchor=(1.24, 0.82))
    style_axes(ax_bottom)
    fig.tight_layout()
    fig.subplots_adjust(right=0.66, hspace=0.42)
    return fig


def _class_target_figure(curve, title_name, samples, exemplars, features, targets, target_name):
    """Feature histograms per target class, with the priority curve on a second axis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    bins = min(50, max(10, int(np.sqrt(features.size))))
    for i, cls in enumerate(np.unique(targets)):
        vals = features[targets == cls]
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.35,
            color=bar_color(i),
            edgecolor=TEXT,
            linewidth=1.0,
            label=f"{target_name}={cls}",
        )
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Class-conditional density")

    ax2 = ax.twinx()
    ax2.plot(
        curve.x,
        curve.y,
        color=line_color("theoretical"),
        linewidth=3.2,
        alpha=0.95,
        label="Priority Function",
    )
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("Priority Weight", color=TEXT)
    ax2.tick_params(axis="y", colors=TEXT)
    draw_pairs_on_curve(
        ax2, curve, samples, exemplars, size=10, edge_width=1.5, names=("sample", "exemplar")
    )

    ax.set_title(f"Priority + class-wise distribution for {title_name}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    styled_legend(ax, h1 + h2, l1 + l2, fontsize=12, anchor=(1.02, 0.9))
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    return fig
