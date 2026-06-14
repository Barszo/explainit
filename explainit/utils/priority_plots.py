"""Reusable plotting helpers for priority functions.

Centralises the matplotlib code that used to live in
``RandomSearchExplainer.display_priorities`` /
``RandomSearchExplainer.investigate_probability_distribution`` and
``MINLSearchExplainer.display_priorities``.

Public entry points
-------------------
Static priority weights (what the optimiser actually sees):
- ``plot_numerical_priority(idx, constraint, ...)`` - one numerical feature.
- ``plot_categorical_priority(group_indices, mapping, ...)`` - one categorical
  group.
- ``plot_priorities(priorities, ...)`` - convenience wrapper that walks the
  full priorities dict and plots every actionable entry.

Probability-distribution views (sampling-oriented interpretation of the same
priorities -- mirrors the old
``RandomSearchExplainer.investigate_probability_distribution``):
- ``plot_numerical_probability_distribution(idx, constraint, ...)`` - theoretical
  density (priority normalised by area) overlaid with the empirical histogram
  obtained via Monte-Carlo rejection sampling.
- ``plot_categorical_probability_distribution(group_indices, mapping, ...)`` -
  bars of normalised probabilities, with forbidden (zero-weight) combinations
  filtered out.
- ``plot_probability_distributions(priorities, ...)`` - walks the priorities
  dict and produces a distribution plot per actionable entry.

Sampling helpers:
- ``sample_numeric_value(constraint, max_tries=100)`` - rejection sampler used
  by the distribution plots. Exposed so callers can reuse the exact same
  sampling logic that the plots rely on.

All functions accept an optional ``save_dir`` (or explicit ``save_path``) so
callers can persist plots to disk without touching matplotlib state.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from explainit.utils.plot_styles import (
    COLORS,
    apply_style,
    get_bar_color,
    get_line_color,
    style_categorical_plot,
    style_numerical_plot,
)


PathLike = Union[str, os.PathLike]


PAIR_COLORS: Tuple[str, ...] = (
    "#FF6B35",
    "#FFD166",
    "#06D6A0",
    "#118AB2",
    "#9D4EDD",
    "#EF476F",
    "#26C485",
    "#F4A261",
)


def _pair_color(i: int) -> str:
    return PAIR_COLORS[i % len(PAIR_COLORS)]


def _normalize_vectors(vectors: Optional[Any]) -> Optional[List[np.ndarray]]:
    """Normalise ``sample``/``exemplar`` to a list of 1D vectors.

    Accepts None, a single 1D vector, or a 2D array / list-of-vectors.
    Returns None if input is None, otherwise a list of 1D ``np.ndarray``.
    """

    if vectors is None:
        return None
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(
        f"Expected 1D or 2D array for sample/exemplar, got ndim={arr.ndim}"
    )


def _normalize_target_values(
    values: Optional[Any], n_pairs: int
) -> Optional[List[Optional[float]]]:
    """Normalise scalar or list-of-scalars target values to length ``n_pairs``.

    Returns ``None`` for inputs that are ``None``. A scalar input is
    broadcast to a length-``n_pairs`` list. Shorter lists are right-padded
    with ``None``; longer lists are truncated.
    """

    if values is None:
        return None
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)] * n_pairs
    arr = list(values)
    out: List[Optional[float]] = []
    for i in range(n_pairs):
        if i >= len(arr) or arr[i] is None:
            out.append(None)
        else:
            try:
                out.append(float(arr[i]))
            except (TypeError, ValueError):
                out.append(None)
    return out


def _pair_label(prefix: str, idx: int, total: int) -> str:
    if total <= 1:
        return prefix
    return f"{prefix} #{idx + 1}"


def _ensure_dir(directory: Optional[PathLike]) -> Optional[Path]:
    if directory is None:
        return None
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_actionable_numeric(constraint: Any) -> bool:
    if not isinstance(constraint, dict) or "function" not in constraint:
        return False
    f = constraint.get("function")
    min_val = constraint.get("min")
    max_val = constraint.get("max")
    if f is None:
        return False
    if min_val is None or max_val is None:
        return False
    if min_val == max_val:
        return False
    return True


def _bar_width_for(num_categories: int) -> float:
    if num_categories == 1:
        return 0.1
    if num_categories == 2:
        return 0.4
    if num_categories <= 4:
        return 0.5
    if num_categories <= 6:
        return 0.6
    return 0.8


def sample_numeric_value(constraint: Mapping[str, Any], max_tries: int = 100) -> float:
    """Rejection-sample a value from the [min, max] range proportional to
    ``constraint['function']`` (which must take values in [0, 1]).

    Falls back to a discretised inverse-transform on a 256-point grid if
    rejection sampling does not succeed within ``max_tries`` iterations.
    """

    min_val = float(constraint["min"])
    max_val = float(constraint["max"])
    f = constraint["function"]

    for _ in range(max_tries):
        rv = np.random.uniform(min_val, max_val)
        w = float(np.asarray(f(rv)).squeeze())
        if np.random.random() < w:
            return float(rv)

    xs = np.linspace(min_val, max_val, 256)
    ws = np.asarray(f(xs)).astype(float).ravel()
    ws = np.clip(ws, 0.0, None)
    if ws.sum() == 0 or not np.isfinite(ws).all():
        return float(np.random.uniform(min_val, max_val))
    p = ws / ws.sum()
    idx = np.random.choice(len(xs), p=p)
    jitter = (max_val - min_val) / 256 * (np.random.random() - 0.5)
    return float(np.clip(xs[idx] + jitter, min_val, max_val))


def _finalize(fig: plt.Figure, save_path: Optional[PathLike], show: bool) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _infer_target_type(target_values: np.ndarray) -> str:
    vals = np.asarray(target_values).flatten()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "unknown"
    uniq = np.unique(vals)
    if uniq.size <= 12:
        return "categorical"
    return "continuous"


def _find_level_x_positions(
    x_vals: np.ndarray, y_vals: np.ndarray, level: float, tol: float = 1e-4
) -> List[float]:
    if x_vals.size < 2 or y_vals.size < 2:
        return []
    out: List[float] = []
    d = y_vals - float(level)
    # For flat runs where priority == level, draw one meaningful boundary:
    # - run touching left edge  -> use RIGHT boundary (border to >/< level)
    # - otherwise               -> use LEFT boundary
    close = np.isclose(d, 0.0, atol=tol)
    if np.any(close):
        starts = np.where(close & np.concatenate(([True], ~close[:-1])))[0]
        ends = np.where(close & np.concatenate((~close[1:], [True])))[0]
        for s, e in zip(starts, ends):
            if s == 0 and e + 1 < len(x_vals):
                out.append(float(x_vals[e]))   # left-edge run -> right boundary
            else:
                out.append(float(x_vals[s]))   # default -> left boundary
    # zero crossings with linear interpolation
    sign = np.sign(d)
    for i in range(len(sign) - 1):
        if sign[i] == 0 or sign[i + 1] == 0:
            continue
        if sign[i] == sign[i + 1]:
            continue
        x0, x1 = float(x_vals[i]), float(x_vals[i + 1])
        y0, y1 = float(y_vals[i]), float(y_vals[i + 1])
        if abs(y1 - y0) < 1e-12:
            continue
        t = (level - y0) / (y1 - y0)
        out.append(x0 + t * (x1 - x0))
    out.sort()
    dedup: List[float] = []
    for x in out:
        if not dedup or abs(x - dedup[-1]) > 2e-3:
            dedup.append(x)
    return dedup


def plot_numerical_priority(
    idx: int,
    constraint: Mapping[str, Any],
    *,
    sample: Optional[Any] = None,
    exemplar: Optional[Any] = None,
    sample_target_value: Optional[Any] = None,
    exemplar_target_value: Optional[Any] = None,
    feature_values: Optional[Sequence[float]] = None,
    target_values: Optional[Sequence[float]] = None,
    target_name: str = "target",
    target_threshold: Optional[float] = None,
    feature_name: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Plot the priority weight curve for a single numerical feature.

    ``sample`` and ``exemplar`` can be either a single 1D feature vector or a
    list/2D-array of such vectors. When multiple pairs are passed, each
    sample/exemplar pair is rendered with a distinct colour and connected
    by a translucent line so the pairing remains visible.

    Returns the figure (or ``None`` if the feature is non-actionable and was
    skipped). When ``save_dir`` is provided a default filename is used.
    """

    if not _is_actionable_numeric(constraint):
        return None

    apply_style()

    min_val = float(constraint["min"])
    max_val = float(constraint["max"])
    f = constraint["function"]

    x_vals = np.linspace(min_val, max_val, 1200)
    priority_values = np.array([float(f(x)) for x in x_vals])
    title_name = feature_name if feature_name else f"Feature {idx}"

    samples_list = _normalize_vectors(sample)
    exemplars_list = _normalize_vectors(exemplar)
    n_pairs = max(
        len(samples_list) if samples_list is not None else 0,
        len(exemplars_list) if exemplars_list is not None else 0,
        0,
    )
    sample_target_list = _normalize_target_values(sample_target_value, n_pairs)
    exemplar_target_list = _normalize_target_values(exemplar_target_value, n_pairs)

    def _val_at(vec_list: Optional[List[np.ndarray]], i: int) -> Optional[float]:
        if vec_list is None or i >= len(vec_list):
            return None
        vec = vec_list[i]
        if idx >= vec.size:
            return None
        v = float(vec[idx])
        return v if np.isfinite(v) else None

    sample_values: List[Optional[float]] = [
        _val_at(samples_list, i) for i in range(n_pairs)
    ]
    exemplar_values: List[Optional[float]] = [
        _val_at(exemplars_list, i) for i in range(n_pairs)
    ]
    # Back-compat shortcuts for legacy single-pair code paths.
    sample_value = sample_values[0] if sample_values else None
    exemplar_value = exemplar_values[0] if exemplar_values else None

    has_target_distribution = feature_values is not None and target_values is not None
    if has_target_distribution:
        feature_arr = np.asarray(feature_values, dtype=float).flatten()
        target_arr = np.asarray(target_values, dtype=float).flatten()
        mask = np.isfinite(feature_arr) & np.isfinite(target_arr)
        feature_arr = feature_arr[mask]
        target_arr = target_arr[mask]
        has_target_distribution = (
            feature_arr.size >= 3 and target_arr.size == feature_arr.size
        )

    if not has_target_distribution:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(
            x_vals, priority_values, label="Priority Function",
            color=get_line_color("theoretical"), linewidth=4, alpha=0.9,
            solid_capstyle="round",
        )
        ax.fill_between(
            x_vals, priority_values, alpha=0.2, color=get_line_color("theoretical")
        )

        for i in range(n_pairs):
            color = _pair_color(i)
            s_val = sample_values[i]
            e_val = exemplar_values[i]
            s_y = e_y = None
            if s_val is not None and min_val <= s_val <= max_val:
                s_y = float(f(s_val))
            if e_val is not None and min_val <= e_val <= max_val:
                e_y = float(f(e_val))
            if s_y is not None and e_y is not None:
                ax.plot(
                    [s_val, e_val], [s_y, e_y],
                    color=color, alpha=0.35, linewidth=2.0, linestyle="--",
                )
            if s_y is not None:
                ax.plot(
                    s_val, s_y, "v", color=color, markersize=12,
                    label=f"{_pair_label('Sample', i, n_pairs)} ({s_val:.3f})",
                    markeredgecolor=COLORS["dirty_white"], markeredgewidth=2,
                )
            if e_y is not None:
                ax.plot(
                    e_val, e_y, "s", color=color, markersize=12,
                    label=f"{_pair_label('Exemplar', i, n_pairs)} ({e_val:.3f})",
                    markeredgecolor=COLORS["dirty_white"], markeredgewidth=2,
                )

        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Priority Weight")
        ax.set_title(f"Priority Function for Numerical {title_name}")
        legend = ax.legend(
            frameon=True, fancybox=True, shadow=True,
            facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
            fontsize=16, loc="center left", bbox_to_anchor=(1.02, 0.9), ncol=1,
        )
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(COLORS["dirty_white"])
        style_numerical_plot(ax)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
    else:
        target_type = _infer_target_type(target_arr)
        fig, ax = plt.subplots(figsize=(12, 8))

        if target_type == "continuous":
            # Restore original priority curve and place it below the density plot
            fig, (ax_top, ax_bottom) = plt.subplots(
                2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [3, 2]}
            )
            feature_unique = np.unique(feature_arr)
            is_discrete_feature = feature_unique.size <= 12

            if is_discrete_feature:
                plt.close(fig)
                ordered_vals = np.sort(feature_unique)
                grouped = [target_arr[np.isclose(feature_arr, v, atol=1e-9)] for v in ordered_vals]
                n_groups = len(ordered_vals)
                # one histogram per feature value + one bottom priority panel
                fig, axes = plt.subplots(
                    n_groups + 1,
                    1,
                    figsize=(16, max(28, 10.4 * (n_groups + 1))),
                    sharex=False,
                )
                hist_axes = axes[:-1]
                ax_bottom = axes[-1]

                # 1D target histograms per feature value (stacked vertically)
                all_vals = np.concatenate([g for g in grouped if g.size > 0]) if grouped else np.array([])
                if all_vals.size > 0:
                    bins = min(50, max(10, int(np.sqrt(all_vals.size))))
                    x_min = float(np.min(all_vals))
                    x_max = float(np.max(all_vals))
                else:
                    bins = 20
                    x_min, x_max = 0.0, 1.0

                # Compute a common y-limit (density max) across all histograms
                y_max = 1.0
                for vals in grouped:
                    if vals.size == 0:
                        continue
                    hist, _ = np.histogram(vals, bins=bins, range=(x_min, x_max), density=True)
                    if hist.size:
                        y_max = max(y_max, float(np.max(hist)))

                for i, (v, vals) in enumerate(zip(ordered_vals, grouped)):
                    ax_i = hist_axes[i]
                    if vals.size == 0:
                        ax_i.text(
                            0.5, 0.5, "No samples", transform=ax_i.transAxes,
                            ha="center", va="center", color=COLORS["dirty_white"]
                        )
                    else:
                        ax_i.hist(
                            vals,
                            bins=bins,
                            range=(x_min, x_max),
                            density=True,
                            alpha=0.55,
                            color=get_bar_color(i),
                            edgecolor=COLORS["dirty_white"],
                            linewidth=1.0,
                        )
                    if target_threshold is not None and np.isfinite(target_threshold):
                        ax_i.axvline(
                            float(target_threshold), color=COLORS["dirty_white"],
                            linewidth=1.8, linestyle="-.",
                            label=f"{target_name} threshold ({float(target_threshold):.3f})",
                        )
                    pv = float(np.asarray(f(v)).squeeze())
                    ax_i.set_xlim(x_min, x_max)
                    ax_i.set_ylim(0.0, y_max * 1.05)
                    ax_i.set_ylabel("density")
                    ax_i.set_title(f"{title_name}={v:.3f}  |  p={pv:.2f}", fontsize=26)
                    if i == n_groups - 1:
                        ax_i.set_xlabel(target_name, labelpad=8)
                    style_numerical_plot(ax_i)
                    ax_i.tick_params(axis="both", labelsize=24)

                # single legend on the first histogram axis
                handles, labels_leg = hist_axes[0].get_legend_handles_labels()
                if handles:
                    legend = hist_axes[0].legend(
                        frameon=True, fancybox=True, shadow=True,
                        facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
                        fontsize=24, loc="center left", bbox_to_anchor=(1.10, 0.85), ncol=1,
                    )
                    legend.get_frame().set_alpha(0.9)
                    for text in legend.get_texts():
                        text.set_color(COLORS["dirty_white"])

                pvals = np.array([float(np.asarray(f(v)).squeeze()) for v in ordered_vals], dtype=float)
                bars = ax_bottom.bar(
                    range(len(ordered_vals)), pvals,
                    color=[get_bar_color(i) for i in range(len(ordered_vals))],
                    edgecolor=COLORS["dirty_white"], linewidth=1.8, alpha=0.85,
                )
                for i, b in enumerate(bars):
                    ax_bottom.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.02,
                        f"p={pvals[i]:.2f}",
                        ha="center",
                        va="bottom",
                        color=COLORS["dirty_white"],
                        fontsize=26,
                    )
                for pi in range(n_pairs):
                    color = _pair_color(pi)
                    s_v = sample_values[pi]
                    e_v = exemplar_values[pi]
                    s_idx = e_idx = None
                    if s_v is not None:
                        m = np.where(np.isclose(ordered_vals, s_v, atol=1e-9))[0]
                        if m.size:
                            s_idx = int(m[0])
                    if e_v is not None:
                        m = np.where(np.isclose(ordered_vals, e_v, atol=1e-9))[0]
                        if m.size:
                            e_idx = int(m[0])
                    if s_idx is not None and e_idx is not None:
                        ax_bottom.plot(
                            [s_idx, e_idx], [pvals[s_idx], pvals[e_idx]],
                            color=color, alpha=0.35, linewidth=2.0, linestyle="--",
                        )
                    if s_idx is not None:
                        ax_bottom.plot(
                            s_idx, pvals[s_idx], "v", color=color,
                            markersize=12, markeredgecolor=COLORS["dirty_white"],
                            markeredgewidth=1.5,
                            label=_pair_label("Sample", pi, n_pairs),
                        )
                    if e_idx is not None:
                        ax_bottom.plot(
                            e_idx, pvals[e_idx], "s", color=color,
                            markersize=12, markeredgecolor=COLORS["dirty_white"],
                            markeredgewidth=1.5,
                            label=_pair_label("Exemplar", pi, n_pairs),
                        )
                ax_bottom.set_xticks(range(len(ordered_vals)))
                ax_bottom.set_xticklabels([f"{v:.3f}" for v in ordered_vals], rotation=30, ha="right")
                ax_bottom.set_ylim(-0.02, 1.02)
                ax_bottom.set_xlabel("Feature Value")
                ax_bottom.set_ylabel("Priority Weight")
                ax_bottom.set_title(f"Priority by observed {title_name} values", fontsize=26)
                ax_bottom.tick_params(axis="both", labelsize=24)
                handles, _ = ax_bottom.get_legend_handles_labels()
                if handles:
                    legend2 = ax_bottom.legend(
                        frameon=True, fancybox=True, shadow=True,
                        facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
                        fontsize=24, loc="center left", bbox_to_anchor=(1.10, 0.82), ncol=1,
                    )
                    legend2.get_frame().set_alpha(0.9)
                    for text in legend2.get_texts():
                        text.set_color(COLORS["dirty_white"])
                style_numerical_plot(ax_bottom)
                plt.tight_layout()
                plt.subplots_adjust(right=0.80, hspace=0.38)
            else:
                # continuous feature -> 2D target/feature density + original priority panel
                bins_x = min(60, max(10, int(np.sqrt(feature_arr.size))))
                bins_y = min(60, max(10, int(np.sqrt(target_arr.size))))
                h = ax_top.hist2d(feature_arr, target_arr, bins=[bins_x, bins_y], cmap="viridis")
                cbar = fig.colorbar(h[3], ax=ax_top)
                cbar.ax.tick_params(colors=COLORS["dirty_white"])
                cbar.set_label("density", color=COLORS["dirty_white"])
                
                is_constant = np.allclose(priority_values, priority_values[0], atol=1e-5)
                levels = [1.0, 0.75, 0.5, 0.25, 0.0]
                level_colors = {
                    1.0: "#90EE90",
                    0.75: "#A0E7E5",
                    0.5: "#FFD166",
                    0.25: "#F4A261",
                    0.0: "#E76F51",
                }
                if not is_constant:
                    for level in levels:
                        x_positions = _find_level_x_positions(x_vals, priority_values, level)
                        first = True
                        for xv in x_positions:
                            ax_top.axvline(
                                xv, color=level_colors[level], linewidth=2.2,
                                linestyle="--", alpha=0.95,
                                label=(f"priority={level:g}" if first else None),
                            )
                            first = False

                if target_threshold is not None and np.isfinite(target_threshold):
                    ax_top.axhline(
                        float(target_threshold),
                        color=COLORS["dirty_white"],
                        linewidth=2.0,
                        linestyle="-.",
                        label=f"{target_name} threshold ({float(target_threshold):.3f})",
                    )

                median_y = float(np.median(target_arr))
                for pi in range(n_pairs):
                    color = _pair_color(pi)
                    s_v = sample_values[pi]
                    e_v = exemplar_values[pi]
                    s_t = (
                        sample_target_list[pi]
                        if sample_target_list is not None else None
                    )
                    e_t = (
                        exemplar_target_list[pi]
                        if exemplar_target_list is not None else None
                    )
                    s_y = s_t if s_t is not None else median_y
                    e_y = e_t if e_t is not None else median_y
                    if (s_v is not None and np.isfinite(s_v)
                            and e_v is not None and np.isfinite(e_v)):
                        ax_top.plot(
                            [s_v, e_v], [s_y, e_y],
                            color=color, alpha=0.35, linewidth=2.0,
                            linestyle="--",
                        )
                    if s_v is not None and np.isfinite(s_v):
                        ax_top.scatter(
                            [s_v], [s_y], marker="v", s=130, color=color,
                            edgecolors=COLORS["dirty_white"], linewidths=1.5,
                            label=f"{_pair_label('sample', pi, n_pairs)} x={s_v:.3f}",
                        )
                    if e_v is not None and np.isfinite(e_v):
                        ax_top.scatter(
                            [e_v], [e_y], marker="s", s=130, color=color,
                            edgecolors=COLORS["dirty_white"], linewidths=1.5,
                            label=f"{_pair_label('exemplar', pi, n_pairs)} x={e_v:.3f}",
                        )

                ax_top.set_ylabel(target_name)
                ax_top.set_xlabel("Feature Value", labelpad=10)
                ax_top.set_title(f"Priority + {title_name} vs {target_name} density")
                legend = ax_top.legend(
                    frameon=True, fancybox=True, shadow=True,
                    facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
                    fontsize=12, loc="center left", bbox_to_anchor=(1.24, 0.9), ncol=1,
                )
                legend.get_frame().set_alpha(0.9)
                for text in legend.get_texts():
                    text.set_color(COLORS["dirty_white"])
                if is_constant:
                    ax_top.text(
                        1.24, 0.18,
                        "Note: priority is constant on this range;\nlevel lines hidden.",
                        transform=ax_top.transAxes, ha="left", va="top",
                        color=COLORS["dirty_white"], fontsize=11,
                    )
                style_numerical_plot(ax_top)
                ax_top.tick_params(axis="x", labelbottom=True)

                # Previous/original priority function plot restored below density
                ax_bottom.plot(
                    x_vals, priority_values, label="Priority Function",
                    color=get_line_color("theoretical"), linewidth=4, alpha=0.9,
                    solid_capstyle="round",
                )
                ax_bottom.fill_between(
                    x_vals, priority_values, alpha=0.2, color=get_line_color("theoretical")
                )
                for pi in range(n_pairs):
                    color = _pair_color(pi)
                    s_v = sample_values[pi]
                    e_v = exemplar_values[pi]
                    s_y = e_y = None
                    if s_v is not None and min_val <= s_v <= max_val:
                        s_y = float(f(s_v))
                    if e_v is not None and min_val <= e_v <= max_val:
                        e_y = float(f(e_v))
                    if s_y is not None and e_y is not None:
                        ax_bottom.plot(
                            [s_v, e_v], [s_y, e_y],
                            color=color, alpha=0.35, linewidth=2.0,
                            linestyle="--",
                        )
                    if s_y is not None:
                        ax_bottom.plot(
                            s_v, s_y, "v", color=color, markersize=10,
                            label=f"{_pair_label('Sample', pi, n_pairs)} ({s_v:.3f})",
                            markeredgecolor=COLORS["dirty_white"], markeredgewidth=1.5,
                        )
                    if e_y is not None:
                        ax_bottom.plot(
                            e_v, e_y, "s", color=color, markersize=10,
                            label=f"{_pair_label('Exemplar', pi, n_pairs)} ({e_v:.3f})",
                            markeredgecolor=COLORS["dirty_white"], markeredgewidth=1.5,
                        )
                # Show the same boundary symbols on the bottom plot
                if not is_constant:
                    for level in levels:
                        x_positions = _find_level_x_positions(x_vals, priority_values, level)
                        first = True
                        for xv in x_positions:
                            ax_bottom.axvline(
                                xv, color=level_colors[level], linewidth=1.8,
                                linestyle="--", alpha=0.9,
                                label=(f"boundary p={level:g}" if first else None),
                            )
                            first = False
                ax_bottom.set_xlabel("Feature Value")
                ax_bottom.set_ylabel("Priority Weight")
                ax_bottom.set_title(f"Priority Function for Numerical {title_name}")
                legend2 = ax_bottom.legend(
                    frameon=True, fancybox=True, shadow=True,
                    facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
                    fontsize=11, loc="center left", bbox_to_anchor=(1.24, 0.82), ncol=1,
                )
                legend2.get_frame().set_alpha(0.9)
                for text in legend2.get_texts():
                    text.set_color(COLORS["dirty_white"])
                style_numerical_plot(ax_bottom)
                plt.tight_layout()
                plt.subplots_adjust(right=0.66, hspace=0.42)
        else:
            # categorical target: class-wise feature distributions + priority function
            classes = np.unique(target_arr)
            bins = min(50, max(10, int(np.sqrt(feature_arr.size))))
            for i, cls in enumerate(classes):
                vals = feature_arr[target_arr == cls]
                if vals.size == 0:
                    continue
                ax.hist(
                    vals, bins=bins, density=True, alpha=0.35,
                    color=get_bar_color(i), edgecolor=COLORS["dirty_white"], linewidth=1.0,
                    label=f"{target_name}={cls}",
                )
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Class-conditional density")

            ax2 = ax.twinx()
            ax2.plot(
                x_vals, priority_values, color=get_line_color("theoretical"),
                linewidth=3.2, alpha=0.95, label="Priority Function",
            )
            ax2.set_ylim(-0.02, 1.02)
            ax2.set_ylabel("Priority Weight", color=COLORS["dirty_white"])
            ax2.tick_params(axis="y", colors=COLORS["dirty_white"])

            for pi in range(n_pairs):
                color = _pair_color(pi)
                s_v = sample_values[pi]
                e_v = exemplar_values[pi]
                s_y = e_y = None
                if s_v is not None and min_val <= s_v <= max_val:
                    s_y = float(f(s_v))
                if e_v is not None and min_val <= e_v <= max_val:
                    e_y = float(f(e_v))
                if s_y is not None and e_y is not None:
                    ax2.plot(
                        [s_v, e_v], [s_y, e_y],
                        color=color, alpha=0.35, linewidth=2.0,
                        linestyle="--",
                    )
                if s_y is not None:
                    ax2.plot(
                        s_v, s_y, "v", color=color, markersize=10,
                        markeredgecolor=COLORS["dirty_white"], markeredgewidth=1.5,
                        label=f"{_pair_label('sample', pi, n_pairs)} ({s_v:.3f})",
                    )
                if e_y is not None:
                    ax2.plot(
                        e_v, e_y, "s", color=color, markersize=10,
                        markeredgecolor=COLORS["dirty_white"], markeredgewidth=1.5,
                        label=f"{_pair_label('exemplar', pi, n_pairs)} ({e_v:.3f})",
                    )

            ax.set_title(f"Priority + class-wise distribution for {title_name}")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            legend = ax.legend(
                h1 + h2, l1 + l2,
                frameon=True, fancybox=True, shadow=True,
                facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
                fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.9), ncol=1,
            )
            legend.get_frame().set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color(COLORS["dirty_white"])
            style_numerical_plot(ax)
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)

    if save_path is None and save_dir is not None:
        directory = _ensure_dir(save_dir)
        slug = feature_name.replace(" ", "_") if feature_name else f"feature_{idx}"
        save_path = directory / f"priority_function_{idx}_{slug}.png"

    _finalize(fig, save_path, show)
    return fig


def plot_categorical_priority(
    group_indices: Tuple[int, ...],
    mapping: Mapping[Tuple[Any, ...], float],
    *,
    sample: Optional[Any] = None,
    exemplar: Optional[Any] = None,
    feature_matrix: Optional[Sequence[Sequence[float]]] = None,
    target_values: Optional[Sequence[float]] = None,
    target_name: str = "target",
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[PathLike] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Plot a bar chart with the priority weights of a categorical group.

    ``sample`` and ``exemplar`` may be a single 1D feature vector or a
    list/2D-array of vectors. Multiple pairs are highlighted with distinct
    colours and a translucent connector arc.
    """

    if not mapping:
        return None

    apply_style()

    categories: List[Tuple[Any, ...]] = list(mapping.keys())
    weights = np.array([0.0 if v is None else float(v) for v in mapping.values()], dtype=float)
    category_labels = [str(cat) for cat in categories]
    num_categories = len(category_labels)
    bar_width = _bar_width_for(num_categories)

    samples_list_cat = _normalize_vectors(sample)
    exemplars_list_cat = _normalize_vectors(exemplar)
    n_pairs_cat = max(
        len(samples_list_cat) if samples_list_cat is not None else 0,
        len(exemplars_list_cat) if exemplars_list_cat is not None else 0,
        0,
    )

    def _combo_for(vec: np.ndarray) -> Optional[Tuple[Any, ...]]:
        try:
            return tuple(float(vec[gi]) for gi in group_indices)
        except (IndexError, TypeError, ValueError):
            return None

    def _combo_idx(combo: Optional[Tuple[Any, ...]]) -> Optional[int]:
        if combo is None:
            return None
        for i, cat in enumerate(categories):
            try:
                if all(
                    math.isclose(float(a), float(b), abs_tol=1e-9)
                    for a, b in zip(combo, cat)
                ):
                    return i
            except (TypeError, ValueError):
                if combo == cat:
                    return i
        return None

    has_target_distribution = feature_matrix is not None and target_values is not None
    if has_target_distribution:
        fm = np.asarray(feature_matrix, dtype=float)
        tv = np.asarray(target_values, dtype=float).flatten()
        has_target_distribution = fm.ndim == 2 and len(group_indices) > 0 and tv.size == fm.shape[0]
    if has_target_distribution:
        combos = [tuple(row[list(group_indices)]) for row in fm]
        target_type = _infer_target_type(tv)
        mask = np.isfinite(tv)
        tv = tv[mask]
        combos = [c for c, keep in zip(combos, mask) if keep]

        fig, ax = plt.subplots(figsize=(12, 8))
        if target_type == "categorical":
            t_classes = np.unique(tv)
            bottoms = np.zeros(num_categories, dtype=float)
            for j, cls in enumerate(t_classes):
                cls_counts = np.zeros(num_categories, dtype=float)
                for i, cat in enumerate(categories):
                    combo_mask = np.array([c == cat for c in combos], dtype=bool)
                    cls_counts[i] = float(np.sum(combo_mask & (tv == cls)))
                ax.bar(
                    np.arange(num_categories), cls_counts, bottom=bottoms, width=bar_width,
                    color=get_bar_color(j), edgecolor=COLORS["dirty_white"], linewidth=1.2,
                    alpha=0.8, label=f"{target_name}={cls}",
                )
                bottoms += cls_counts
            for i, total in enumerate(bottoms):
                ax.text(
                    i, total + max(bottoms) * 0.02 if max(bottoms) > 0 else 0.1,
                    f"p={weights[i]:.3f}",
                    ha="center", va="bottom", fontsize=11, color=COLORS["dirty_white"],
                )
            ax.set_ylabel("count")
            ax.set_title(f"{target_name} counts by feature group")
        else:
            grouped = []
            labels = []
            for i, cat in enumerate(categories):
                combo_mask = np.array([c == cat for c in combos], dtype=bool)
                vals = tv[combo_mask]
                if vals.size == 0:
                    vals = np.array([np.nan])
                grouped.append(vals)
                labels.append(f"{category_labels[i]}\np={weights[i]:.3f}")
            bp = ax.boxplot(grouped, patch_artist=True, labels=labels)
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(get_bar_color(i))
                box.set_edgecolor(COLORS["dirty_white"])
                box.set_alpha(0.75)
            for med in bp["medians"]:
                med.set_color(COLORS["dirty_white"])
                med.set_linewidth(2.0)
            ax.set_ylabel(target_name)
            ax.set_title(f"{target_name} distribution by feature group")
        ax.set_xlabel("Feature group values")
        if target_type == "categorical":
            ax.set_xticks(range(num_categories))
            ax.set_xticklabels(category_labels, rotation=45, ha="right")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                frameon=True, fancybox=True, shadow=True,
                facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
                fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.9), ncol=1,
            )
            legend.get_frame().set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color(COLORS["dirty_white"])
        style_categorical_plot(ax, num_categories)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        if save_path is None and save_dir is not None:
            directory = _ensure_dir(save_dir)
            group_str = "_".join(map(str, group_indices))
            save_path = directory / f"priority_categorical_{group_str}.png"
        _finalize(fig, save_path, show)
        return fig

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_colors = [get_bar_color(i) for i in range(num_categories)]
    bars = ax.bar(
        range(num_categories), weights, width=bar_width,
        color=bar_colors, edgecolor=COLORS["dirty_white"], linewidth=2.0,
    )
    for i, bar in enumerate(bars):
        bar.set_alpha(0.7 + 0.2 * (i % 2))
        height = bar.get_height()
        if height > 0:
            ax.add_patch(plt.Rectangle(
                (bar.get_x() + 0.02, 0.01),
                bar.get_width() - 0.04, height - 0.02,
                fill=False, edgecolor=COLORS["steel_gray"],
                linewidth=0.8, alpha=0.6,
            ))

    weights_max = float(max(weights)) if num_categories else 0.0
    sample_indices_by_pair: List[Optional[int]] = []
    exemplar_indices_by_pair: List[Optional[int]] = []
    for pi in range(n_pairs_cat):
        color = _pair_color(pi)
        s_vec = samples_list_cat[pi] if samples_list_cat is not None and pi < len(samples_list_cat) else None
        e_vec = exemplars_list_cat[pi] if exemplars_list_cat is not None and pi < len(exemplars_list_cat) else None
        s_idx = _combo_idx(_combo_for(s_vec)) if s_vec is not None else None
        e_idx = _combo_idx(_combo_for(e_vec)) if e_vec is not None else None
        sample_indices_by_pair.append(s_idx)
        exemplar_indices_by_pair.append(e_idx)
        if s_idx is not None and e_idx is not None and s_idx != e_idx:
            y_top = max(weights[s_idx], weights[e_idx]) + (weights_max * 0.18 + 0.05)
            ax.plot(
                [s_idx, e_idx], [y_top, y_top],
                color=color, alpha=0.35, linewidth=2.0, linestyle="--",
            )
        if s_idx is not None:
            bars[s_idx].set_edgecolor(color)
            bars[s_idx].set_linewidth(4)
            ax.plot(
                s_idx, weights[s_idx] + (weights_max * 0.05 + 0.02), "v",
                color=color, markersize=15,
                markeredgecolor=COLORS["dirty_white"], markeredgewidth=2,
                label=_pair_label("Sample", pi, n_pairs_cat),
            )
        if e_idx is not None:
            bars[e_idx].set_edgecolor(color)
            bars[e_idx].set_linewidth(4)
            ax.plot(
                e_idx, weights[e_idx] + (weights_max * 0.12 + 0.04), "s",
                color=color, markersize=15,
                markeredgecolor=COLORS["dirty_white"], markeredgewidth=2,
                label=_pair_label("Exemplar", pi, n_pairs_cat),
            )

    if num_categories == 1:
        ax.set_xlim(-1, 1)

    current_indices = {i for i in sample_indices_by_pair if i is not None}
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        label_text = f"{weight:.3f}"
        if i in current_indices:
            label_text += " (Current)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (weights_max * 0.01 if weights_max else 0.01),
            label_text, ha="center", va="bottom", fontsize=16,
            fontweight="bold", color=COLORS["dirty_white"],
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor=COLORS["dark_background"],
                alpha=0.8, edgecolor=COLORS["dirty_white"], linewidth=1.5,
            ),
        )

    if feature_names is not None:
        names = ", ".join(
            feature_names[i] for i in group_indices if i < len(feature_names)
        )
        title = f"Priority Weights for Categorical Features [{names}]"
    else:
        title = f"Priority Weights for Categorical Features {group_indices}"

    ax.set_xlabel("Category Combinations")
    ax.set_ylabel("Priority Weight")
    ax.set_title(title)
    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(category_labels, rotation=45, ha="right")

    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles, labels_leg,
            frameon=True, fancybox=True, shadow=True,
            facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
            fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.9), ncol=1,
        )
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(COLORS["dirty_white"])

    style_categorical_plot(ax, num_categories)
    plt.tight_layout()
    if handles:
        plt.subplots_adjust(right=0.75)

    if save_path is None and save_dir is not None:
        directory = _ensure_dir(save_dir)
        group_str = "_".join(map(str, group_indices))
        save_path = directory / f"priority_categorical_{group_str}.png"

    _finalize(fig, save_path, show)
    return fig


def plot_priorities(
    priorities: Mapping[str, Any],
    *,
    sample: Optional[Any] = None,
    exemplar: Optional[Any] = None,
    sample_target_value: Optional[Any] = None,
    exemplar_target_value: Optional[Any] = None,
    feature_matrix: Optional[Sequence[Sequence[float]]] = None,
    target_values: Optional[Sequence[float]] = None,
    target_name: str = "target",
    target_threshold: Optional[float] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = False,
) -> List[Path]:
    """Plot every actionable priority in a priorities dict.

    ``sample`` and ``exemplar`` accept either a single 1D vector or a list /
    2D array of vectors so several sample/exemplar pairs can be overlaid on
    each plot.

    Returns the list of paths written when ``save_dir`` is provided.
    """

    written: List[Path] = []
    directory = _ensure_dir(save_dir)

    numerical: Mapping[int, Any] = priorities.get("numerical", {}) or {}
    categorical: Mapping[Tuple[int, ...], Any] = priorities.get("categorical", {}) or {}

    for idx, constraint in numerical.items():
        if not _is_actionable_numeric(constraint):
            continue
        feature_name = (
            feature_names[idx]
            if feature_names is not None and idx < len(feature_names)
            else None
        )
        save_path = None
        if directory is not None:
            slug = (feature_name or f"feature_{idx}").replace(" ", "_")
            save_path = directory / f"priority_function_{idx}_{slug}.png"
        column = None
        if feature_matrix is not None:
            fm = np.asarray(feature_matrix)
            if fm.ndim == 2 and idx < fm.shape[1]:
                column = fm[:, idx]

        plot_numerical_priority(
            idx, constraint,
            sample=sample, exemplar=exemplar,
            sample_target_value=sample_target_value,
            exemplar_target_value=exemplar_target_value,
            feature_values=column,
            target_values=target_values,
            target_name=target_name,
            target_threshold=target_threshold,
            feature_name=feature_name,
            save_path=save_path, show=show,
        )
        if save_path is not None:
            written.append(save_path)

    for group_indices, mapping in categorical.items():
        if not mapping:
            continue
        save_path = None
        if directory is not None:
            group_str = "_".join(map(str, group_indices))
            save_path = directory / f"priority_categorical_{group_str}.png"
        plot_categorical_priority(
            group_indices, mapping,
            sample=sample,
            exemplar=exemplar,
            feature_matrix=feature_matrix,
            target_values=target_values,
            target_name=target_name,
            feature_names=feature_names,
            save_path=save_path, show=show,
        )
        if save_path is not None:
            written.append(save_path)

    return written


def plot_numerical_probability_distribution(
    idx: int,
    constraint: Mapping[str, Any],
    *,
    n_samples: int = 10000,
    sampler: Optional[Callable[[Mapping[str, Any]], float]] = None,
    feature_name: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Plot the theoretical density vs the empirical Monte-Carlo histogram for a
    numerical feature's priority.

    The theoretical curve is the priority function normalised by its integral
    over [min, max]. The empirical histogram is built by repeatedly calling
    ``sampler(constraint)`` (default: :func:`sample_numeric_value`).
    """

    if not _is_actionable_numeric(constraint):
        return None

    apply_style()

    min_val = float(constraint["min"])
    max_val = float(constraint["max"])
    f = constraint["function"]
    draw = sampler if sampler is not None else sample_numeric_value

    x_vals = np.linspace(min_val, max_val, 1000)
    weights = np.array([float(f(x)) for x in x_vals])
    area = float(np.trapezoid(weights, x_vals))
    if area <= 0 or not np.isfinite(area):
        prob_density = np.zeros_like(weights)
    else:
        prob_density = weights / area

    samples = np.array([draw(constraint) for _ in range(int(n_samples))])
    hist, bin_edges = np.histogram(samples, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        x_vals, prob_density, label="Theoretical Distribution",
        color=get_line_color("theoretical"), linewidth=4, alpha=0.9,
        solid_capstyle="round",
    )
    ax.fill_between(x_vals, prob_density, alpha=0.2, color=get_line_color("theoretical"))

    bars = ax.bar(
        bin_centers, hist, width=np.diff(bin_edges), alpha=0.8,
        label="Empirical Distribution", color=get_line_color("empirical"),
        edgecolor=COLORS["dirty_white"], linewidth=1.5,
    )
    for i, bar in enumerate(bars):
        bar.set_alpha(0.6 + 0.3 * (i % 2 == 0))

    title_name = feature_name if feature_name else f"Feature {idx}"
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Probability Distribution for Numerical {title_name}")

    legend = ax.legend(
        frameon=True, fancybox=True, shadow=True,
        facecolor=COLORS["dark_background"], edgecolor=COLORS["dirty_white"],
        fontsize=16, loc="center left", bbox_to_anchor=(1.02, 0.9), ncol=1,
    )
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(COLORS["dirty_white"])

    style_numerical_plot(ax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    if save_path is None and save_dir is not None:
        directory = _ensure_dir(save_dir)
        slug = (feature_name or f"feature_{idx}").replace(" ", "_")
        save_path = directory / f"probability_distribution_{idx}_{slug}.png"

    _finalize(fig, save_path, show)
    return fig


def plot_categorical_probability_distribution(
    group_indices: Tuple[int, ...],
    mapping: Mapping[Tuple[Any, ...], float],
    *,
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[PathLike] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Plot normalised sampling probabilities for a categorical group.

    Forbidden combinations (weight == 0 or None) are filtered out before
    renormalising. Returns ``None`` if no combination survives.
    """

    if not mapping:
        return None

    apply_style()

    categories = list(mapping.keys())
    weights = np.array(
        [0.0 if v is None else float(v) for v in mapping.values()], dtype=float,
    )
    allowed_mask = weights > 0
    if not np.any(allowed_mask):
        return None

    allowed_categories = [categories[i] for i in range(len(categories)) if allowed_mask[i]]
    allowed_weights = weights[allowed_mask]
    probabilities = allowed_weights / allowed_weights.sum()
    category_labels = [str(cat) for cat in allowed_categories]
    num_categories = len(category_labels)
    bar_width = _bar_width_for(num_categories)

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_colors = [get_bar_color(i) for i in range(num_categories)]
    bars = ax.bar(
        range(num_categories), probabilities, width=bar_width,
        color=bar_colors, edgecolor=COLORS["dirty_white"], linewidth=2.0,
    )
    for i, bar in enumerate(bars):
        bar.set_alpha(0.7 + 0.2 * (i % 2))
        height = bar.get_height()
        if height > 0:
            ax.add_patch(plt.Rectangle(
                (bar.get_x() + 0.02, 0.01),
                bar.get_width() - 0.04, height - 0.02,
                fill=False, edgecolor=COLORS["steel_gray"],
                linewidth=0.8, alpha=0.6,
            ))

    if num_categories == 1:
        ax.set_xlim(-1, 1)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{prob:.3f}", ha="center", va="bottom", fontsize=16,
            fontweight="bold", color=COLORS["dirty_white"],
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor=COLORS["dark_background"],
                alpha=0.8, edgecolor=COLORS["dirty_white"], linewidth=1.5,
            ),
        )

    if feature_names is not None:
        names = ", ".join(
            feature_names[i] for i in group_indices if i < len(feature_names)
        )
        title = f"Probability Distribution for Categorical Features [{names}]"
    else:
        title = f"Probability Distribution for Categorical Features {group_indices}"

    ax.set_xlabel("Category Combinations")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(category_labels, rotation=45, ha="right")

    style_categorical_plot(ax, num_categories)
    plt.tight_layout()

    if save_path is None and save_dir is not None:
        directory = _ensure_dir(save_dir)
        group_str = "_".join(map(str, group_indices))
        save_path = directory / f"probability_distribution_categorical_{group_str}.png"

    _finalize(fig, save_path, show)
    return fig


def plot_probability_distributions(
    priorities: Mapping[str, Any],
    *,
    n_samples: int = 10000,
    sampler: Optional[Callable[[Mapping[str, Any]], float]] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = False,
) -> List[Path]:
    """Render probability-distribution plots for every actionable entry in
    ``priorities``. Mirrors ``plot_priorities`` but uses the sampling-oriented
    views.
    """

    written: List[Path] = []
    directory = _ensure_dir(save_dir)

    numerical: Mapping[int, Any] = priorities.get("numerical", {}) or {}
    categorical: Mapping[Tuple[int, ...], Any] = priorities.get("categorical", {}) or {}

    for idx, constraint in numerical.items():
        if not _is_actionable_numeric(constraint):
            continue
        feature_name = (
            feature_names[idx]
            if feature_names is not None and idx < len(feature_names)
            else None
        )
        save_path = None
        if directory is not None:
            slug = (feature_name or f"feature_{idx}").replace(" ", "_")
            save_path = directory / f"probability_distribution_{idx}_{slug}.png"
        plot_numerical_probability_distribution(
            idx, constraint,
            n_samples=n_samples, sampler=sampler,
            feature_name=feature_name,
            save_path=save_path, show=show,
        )
        if save_path is not None:
            written.append(save_path)

    for group_indices, mapping in categorical.items():
        if not mapping:
            continue
        save_path = None
        if directory is not None:
            group_str = "_".join(map(str, group_indices))
            save_path = directory / f"probability_distribution_categorical_{group_str}.png"
        plot_categorical_probability_distribution(
            group_indices, mapping,
            feature_names=feature_names,
            save_path=save_path, show=show,
        )
        if save_path is not None:
            written.append(save_path)

    return written


__all__ = [
    "sample_numeric_value",
    "plot_numerical_priority",
    "plot_categorical_priority",
    "plot_priorities",
    "plot_numerical_probability_distribution",
    "plot_categorical_probability_distribution",
    "plot_probability_distributions",
]
