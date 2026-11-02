"""How the random search samples each feature.

Numerical features: the priority normalised to a probability density
(theoretical) against a histogram of values drawn by the sampler (empirical).
Categorical groups: the sampling probability of each allowed combination.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from explainit.plotting._common import (
    PriorityCurve,
    draw_bar_label,
    draw_bars,
    ensure_dir,
    finalize,
    group_path,
    group_title,
    is_actionable_numeric,
    numerical_path,
)
from explainit.plotting.style import TEXT, apply_style, line_color, style_axes, styled_legend
from explainit.sampling import sample_numeric_value
from explainit.typing import PathLike


def plot_numerical_probability_distribution(
    idx: int,
    constraint: Mapping[str, Any],
    *,
    n_samples: int = 10000,
    sampler: Callable[[Mapping[str, Any]], float] | None = None,
    feature_name: str | None = None,
    save_path: PathLike | None = None,
    save_dir: PathLike | None = None,
    show: bool = False,
) -> Figure | None:
    """Theoretical vs. empirical sampling density of numerical feature ``idx``.

    The theoretical curve is the priority divided by its integral over
    ``[min, max]``; the histogram comes from ``n_samples`` calls to
    ``sampler(constraint)`` (default: :func:`explainit.sampling.sample_numeric_value`).

    Returns:
        The figure, or ``None`` if the feature is immutable.
    """
    if not is_actionable_numeric(constraint):
        return None

    apply_style()
    curve = PriorityCurve.from_constraint(constraint, points=1000)
    draw = sampler if sampler is not None else sample_numeric_value
    area = float(np.trapezoid(curve.y, curve.x))
    density = np.zeros_like(curve.y) if area <= 0 or not np.isfinite(area) else curve.y / area

    samples = np.array([draw(constraint) for _ in range(int(n_samples))])
    hist, bin_edges = np.histogram(samples, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        curve.x,
        density,
        label="Theoretical Distribution",
        color=line_color("theoretical"),
        linewidth=4,
        alpha=0.9,
        solid_capstyle="round",
    )
    ax.fill_between(curve.x, density, alpha=0.2, color=line_color("theoretical"))
    bars = ax.bar(
        bin_centers,
        hist,
        width=np.diff(bin_edges),
        alpha=0.8,
        label="Empirical Distribution",
        color=line_color("empirical"),
        edgecolor=TEXT,
        linewidth=1.5,
    )
    for i, bar in enumerate(bars):
        bar.set_alpha(0.6 + 0.3 * (i % 2 == 0))

    title_name = feature_name if feature_name else f"Feature {idx}"
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Probability Distribution for Numerical {title_name}")
    styled_legend(ax, fontsize=16, anchor=(1.02, 0.9))
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    if save_path is None and save_dir is not None:
        save_path = numerical_path(
            ensure_dir(save_dir), "probability_distribution", idx, feature_name
        )
    finalize(fig, save_path, show)
    return fig


def plot_categorical_probability_distribution(
    group_indices: tuple[int, ...],
    mapping: Mapping[tuple[Any, ...], float | None],
    *,
    feature_names: Sequence[str] | None = None,
    save_path: PathLike | None = None,
    save_dir: PathLike | None = None,
    show: bool = False,
) -> Figure | None:
    """Sampling probability of each allowed combination of a categorical group.

    Combinations with weight ``0`` or ``None`` are never sampled and are left
    out.

    Returns:
        The figure, or ``None`` if no combination is allowed.
    """
    if not mapping:
        return None

    apply_style()
    categories = list(mapping.keys())
    weights = np.array([0.0 if v is None else float(v) for v in mapping.values()], dtype=float)
    allowed = weights > 0
    if not np.any(allowed):
        return None
    labels = [str(categories[i]) for i in range(len(categories)) if allowed[i]]
    probabilities = weights[allowed] / weights[allowed].sum()

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = draw_bars(ax, probabilities)
    if len(labels) == 1:
        ax.set_xlim(-1, 1)
    for bar, probability in zip(bars, probabilities, strict=False):
        draw_bar_label(ax, bar, f"{probability:.3f}", 0.003)

    ax.set_xlabel("Category Combinations")
    ax.set_ylabel("Probability")
    ax.set_title(group_title("Probability Distribution", group_indices, feature_names))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    style_axes(ax, grid="y")
    fig.tight_layout()

    if save_path is None and save_dir is not None:
        save_path = group_path(
            ensure_dir(save_dir), "probability_distribution_categorical", group_indices
        )
    finalize(fig, save_path, show)
    return fig


def plot_probability_distributions(
    priorities: Mapping[str, Any],
    *,
    n_samples: int = 10000,
    sampler: Callable[[Mapping[str, Any]], float] | None = None,
    feature_names: Sequence[str] | None = None,
    save_dir: PathLike | None = None,
    show: bool = False,
) -> list[Path]:
    """Sampling distribution plots for every mutable feature in ``priorities``.

    Returns:
        The files written (empty unless ``save_dir`` is given).
    """
    written: list[Path] = []
    directory = ensure_dir(save_dir)

    for idx, constraint in (priorities.get("numerical", {}) or {}).items():
        if not is_actionable_numeric(constraint):
            continue
        name = (
            feature_names[idx] if feature_names is not None and idx < len(feature_names) else None
        )
        path = (
            numerical_path(directory, "probability_distribution", idx, name)
            if directory is not None
            else None
        )
        plot_numerical_probability_distribution(
            idx,
            constraint,
            n_samples=n_samples,
            sampler=sampler,
            feature_name=name,
            save_path=path,
            show=show,
        )
        if path is not None:
            written.append(path)

    for group_indices, mapping in (priorities.get("categorical", {}) or {}).items():
        if not mapping:
            continue
        path = (
            group_path(directory, "probability_distribution_categorical", group_indices)
            if directory is not None
            else None
        )
        plot_categorical_probability_distribution(
            group_indices, mapping, feature_names=feature_names, save_path=path, show=show
        )
        if path is not None:  # also when nothing was drawn: known issue PLOT-3
            written.append(path)

    return written
