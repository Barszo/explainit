"""Helpers shared by the plot modules."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from explainit.plotting.style import (
    COLORS,
    LEVEL_COLORS,
    TEXT,
    VALUE_LABEL_BOX,
    bar_color,
    line_color,
    pair_color,
)

PathLike = str | os.PathLike


# --- inputs -------------------------------------------------------------------


def normalize_vectors(vectors: Any | None) -> list[np.ndarray] | None:
    """``None``, one 1-D vector, or a 2-D array / list of vectors -> list of 1-D vectors."""
    if vectors is None:
        return None
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Expected 1D or 2D array for sample/exemplar, got ndim={arr.ndim}")


def pair_count(samples: list | None, exemplars: list | None) -> int:
    return max(
        len(samples) if samples is not None else 0,
        len(exemplars) if exemplars is not None else 0,
        0,
    )


def normalize_target_values(values: Any | None, n_pairs: int) -> list[float | None] | None:
    """A scalar (broadcast) or a list (padded with ``None``/truncated) -> ``n_pairs`` values."""
    if values is None:
        return None
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)] * n_pairs
    arr = list(values)
    out: list[float | None] = []
    for i in range(n_pairs):
        if i >= len(arr) or arr[i] is None:
            out.append(None)
        else:
            try:
                out.append(float(arr[i]))
            except (TypeError, ValueError):
                out.append(None)
    return out


def values_at(vectors: list[np.ndarray] | None, n_pairs: int, idx: int) -> list:
    """Finite value of feature ``idx`` in each of ``n_pairs`` vectors (``None`` when missing)."""
    out: list[float | None] = []
    for i in range(n_pairs):
        if vectors is None or i >= len(vectors) or idx >= vectors[i].size:
            out.append(None)
            continue
        v = float(vectors[i][idx])
        out.append(v if np.isfinite(v) else None)
    return out


def pair_label(prefix: str, idx: int, total: int) -> str:
    if total <= 1:
        return prefix
    return f"{prefix} #{idx + 1}"


def is_actionable_numeric(constraint: Any) -> bool:
    """A numerical priority with a function and a non-empty ``[min, max]`` range."""
    if not isinstance(constraint, dict) or "function" not in constraint:
        return False
    if constraint.get("function") is None:
        return False
    min_val = constraint.get("min")
    max_val = constraint.get("max")
    if min_val is None or max_val is None:
        return False
    return min_val != max_val


def infer_target_type(target_values: np.ndarray) -> str:
    """``"categorical"`` (at most 12 distinct values), ``"continuous"`` or ``"unknown"``."""
    vals = np.asarray(target_values).flatten()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "unknown"
    if np.unique(vals).size <= 12:
        return "categorical"
    return "continuous"


@dataclass(frozen=True)
class PriorityCurve:
    """A numerical priority function evaluated on a grid over ``[low, high]``."""

    low: float
    high: float
    function: Any
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_constraint(cls, constraint, points: int) -> "PriorityCurve":
        low, high = float(constraint["min"]), float(constraint["max"])
        f = constraint["function"]
        x = np.linspace(low, high, points)
        return cls(low, high, f, x, np.array([float(f(v)) for v in x]))

    def at(self, value: float | None) -> float | None:
        """Priority at ``value``, or ``None`` outside ``[low, high]``."""
        if value is None or not self.low <= value <= self.high:
            return None
        return float(self.function(value))

    @property
    def is_constant(self) -> bool:
        return bool(np.allclose(self.y, self.y[0], atol=1e-5))


# --- drawing ------------------------------------------------------------------


def draw_curve(ax, curve: PriorityCurve) -> None:
    """The priority curve, filled below."""
    ax.plot(
        curve.x,
        curve.y,
        label="Priority Function",
        color=line_color("theoretical"),
        linewidth=4,
        alpha=0.9,
        solid_capstyle="round",
    )
    ax.fill_between(curve.x, curve.y, alpha=0.2, color=line_color("theoretical"))


def draw_connector(ax, x, y, color) -> None:
    """A translucent dashed line joining a sample to its exemplar."""
    ax.plot(x, y, color=color, alpha=0.35, linewidth=2.0, linestyle="--")


def draw_marker(ax, x, y, marker, color, *, size, edge_width, label) -> None:
    ax.plot(
        x,
        y,
        marker,
        color=color,
        markersize=size,
        label=label,
        markeredgecolor=TEXT,
        markeredgewidth=edge_width,
    )


def draw_pairs_on_curve(
    ax, curve, samples, exemplars, *, size, edge_width, names=("Sample", "Exemplar")
) -> None:
    """Mark each sample (triangle) and exemplar (square) on the priority curve."""
    n_pairs = len(samples)
    for i in range(n_pairs):
        color = pair_color(i)
        s_val, e_val = samples[i], exemplars[i]
        s_y, e_y = curve.at(s_val), curve.at(e_val)
        if s_y is not None and e_y is not None:
            draw_connector(ax, [s_val, e_val], [s_y, e_y], color)
        if s_y is not None:
            label = f"{pair_label(names[0], i, n_pairs)} ({s_val:.3f})"
            draw_marker(ax, s_val, s_y, "v", color, size=size, edge_width=edge_width, label=label)
        if e_y is not None:
            label = f"{pair_label(names[1], i, n_pairs)} ({e_val:.3f})"
            draw_marker(ax, e_val, e_y, "s", color, size=size, edge_width=edge_width, label=label)


def level_positions(x_vals: np.ndarray, y_vals: np.ndarray, level: float, tol: float = 1e-4):
    """Feature values where the priority curve reaches ``level``.

    Interpolated crossings, plus one boundary of every flat run at the level
    (the right end for a run touching the left edge, otherwise the left end).
    """
    if x_vals.size < 2 or y_vals.size < 2:
        return []
    out: list[float] = []
    d = y_vals - float(level)
    close = np.isclose(d, 0.0, atol=tol)
    if np.any(close):
        starts = np.where(close & np.concatenate(([True], ~close[:-1])))[0]
        ends = np.where(close & np.concatenate((~close[1:], [True])))[0]
        for s, e in zip(starts, ends, strict=False):
            if s == 0 and e + 1 < len(x_vals):
                out.append(float(x_vals[e]))
            else:
                out.append(float(x_vals[s]))
    sign = np.sign(d)
    for i in range(len(sign) - 1):
        if sign[i] == 0 or sign[i + 1] == 0 or sign[i] == sign[i + 1]:
            continue
        x0, x1 = float(x_vals[i]), float(x_vals[i + 1])
        y0, y1 = float(y_vals[i]), float(y_vals[i + 1])
        if abs(y1 - y0) < 1e-12:
            continue
        t = (level - y0) / (y1 - y0)
        out.append(x0 + t * (x1 - x0))
    out.sort()
    dedup: list[float] = []
    for x in out:
        if not dedup or abs(x - dedup[-1]) > 2e-3:
            dedup.append(x)
    return dedup


def draw_level_lines(ax, curve: PriorityCurve, *, linewidth, alpha, label) -> None:
    """Vertical lines where the priority is 1, 0.75, 0.5, 0.25 and 0."""
    for level, color in LEVEL_COLORS.items():
        first = True
        for xv in level_positions(curve.x, curve.y, level):
            ax.axvline(
                xv,
                color=color,
                linewidth=linewidth,
                linestyle="--",
                alpha=alpha,
                label=(f"{label}={level:g}" if first else None),
            )
            first = False


def bar_width(num_categories: int) -> float:
    if num_categories == 1:
        return 0.1
    if num_categories == 2:
        return 0.4
    if num_categories <= 4:
        return 0.5
    if num_categories <= 6:
        return 0.6
    return 0.8


def draw_bars(ax, heights):
    """Themed bars with an inner outline, one per category."""
    num_categories = len(heights)
    bars = ax.bar(
        range(num_categories),
        heights,
        width=bar_width(num_categories),
        color=[bar_color(i) for i in range(num_categories)],
        edgecolor=TEXT,
        linewidth=2.0,
    )
    for i, bar in enumerate(bars):
        bar.set_alpha(0.7 + 0.2 * (i % 2))
        height = bar.get_height()
        if height > 0:
            ax.add_patch(
                plt.Rectangle(
                    (bar.get_x() + 0.02, 0.01),
                    bar.get_width() - 0.04,
                    height - 0.02,
                    fill=False,
                    edgecolor=COLORS["steel_gray"],
                    linewidth=0.8,
                    alpha=0.6,
                )
            )
    return bars


def draw_bar_label(ax, bar, text, offset) -> None:
    """A boxed value label above ``bar``."""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        text,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
        bbox=VALUE_LABEL_BOX,
    )


def group_title(prefix: str, group_indices, feature_names) -> str:
    if feature_names is not None:
        names = ", ".join(feature_names[i] for i in group_indices if i < len(feature_names))
        return f"{prefix} for Categorical Features [{names}]"
    return f"{prefix} for Categorical Features {group_indices}"


# --- output -------------------------------------------------------------------


def ensure_dir(directory: PathLike | None) -> Path | None:
    if directory is None:
        return None
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def numerical_path(directory: Path, prefix: str, idx: int, feature_name: str | None) -> Path:
    slug = (feature_name or f"feature_{idx}").replace(" ", "_")
    return directory / f"{prefix}_{idx}_{slug}.png"


def group_path(directory: Path, prefix: str, group_indices) -> Path:
    return directory / f"{prefix}_{'_'.join(map(str, group_indices))}.png"


def finalize(fig, save_path: PathLike | None, show: bool) -> None:
    """Save ``fig`` if a path is given, then show it or close it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
