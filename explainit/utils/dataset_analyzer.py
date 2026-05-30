"""Dataset analysis utilities.

Given a feature matrix ``X`` and a target vector ``y`` (and optionally a
sample/exemplar of interest), this module performs a multi-stage diagnostic:

1. Sanity-check the dataset (non-empty, sufficient samples, no all-NaN
   columns, target/feature alignment).
2. Classify the target as ``binary``, ``categorical`` or ``continuous`` and
   describe it (distribution, ranges, balance, correlation with features).
3. For each feature, infer its type (``binary``, ``categorical``,
   ``continuous``), check validity (constant, missing values, very skewed),
   compute descriptive statistics and produce a distribution plot. When a
   ``sample`` and/or ``exemplar`` is supplied, both points are overlaid on
   every relevant plot for visual context.
4. Produce a feature-feature correlation heatmap and a feature-vs-target
   summary plot.
5. Save a plain-text summary describing all findings.

The module is intentionally framework-agnostic: it accepts ``pandas`` data
frames, NumPy arrays or plain lists and returns a structured
``DatasetReport`` describing the analysis. All plots reuse the dark-theme
styling defined in ``explainit.utils.plot_styles``.

Typical usage::

    from explainit.utils.dataset_analyzer import analyze_dataset
    report = analyze_dataset(
        X=ctx.X_train, y=ctx.y_train,
        feature_names=ctx.feature_names,
        dataset_key=ctx.dataset_key,
        output_dir=Path("development/images") / f"{ctx.dataset_key}_analysis",
        sample=sample, exemplar=exemplar,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from explainit.utils.plot_styles import (
    COLORS,
    apply_style,
    get_bar_color,
    get_line_color,
    style_categorical_plot,
    style_numerical_plot,
)


logger = logging.getLogger(__name__)


T_BINARY = "binary"
T_CATEGORICAL = "categorical"
T_CONTINUOUS = "continuous"
T_CONSTANT = "constant"
T_UNKNOWN = "unknown"


_SAMPLE_COLOR = "#FF6B35"
_EXEMPLAR_COLOR = COLORS["moss_green"]


# ---------------------------------------------------------------------------
# Dataclasses describing the analysis
# ---------------------------------------------------------------------------


@dataclass
class FeatureSummary:
    index: int
    name: str
    inferred_type: str
    is_valid: bool
    n_unique: int
    n_missing: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    top_categories: Optional[List[Tuple[Any, int]]] = None
    correlation_with_target: Optional[float] = None
    correlation_kind: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class TargetSummary:
    inferred_type: str
    is_valid: bool
    n_unique: int
    n_missing: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    class_counts: Optional[List[Tuple[Any, int]]] = None
    class_balance: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    dataset_key: str
    n_samples: int
    n_features: int
    feature_names: List[str]
    target_name: str
    is_valid: bool
    issues: List[str]
    target: Optional[TargetSummary]
    features: List[FeatureSummary]
    saved_plots: List[Path] = field(default_factory=list)
    summary_text_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_2d_array(X: Any) -> Tuple[np.ndarray, List[str]]:
    if _HAS_PANDAS and isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        return X.to_numpy(dtype=float, copy=True), names
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError):
        arr = np.asarray(arr, dtype=object)
    names = [f"feature_{i}" for i in range(arr.shape[1])]
    return arr, names


def _to_1d_array(y: Any) -> np.ndarray:
    if _HAS_PANDAS and isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    arr = np.asarray(y).flatten()
    return arr


def _infer_target_name(y: Any, explicit_target_name: Optional[str]) -> str:
    if explicit_target_name is not None and str(explicit_target_name).strip():
        return str(explicit_target_name).strip()
    if _HAS_PANDAS and isinstance(y, pd.Series):
        if y.name is not None and str(y.name).strip():
            return str(y.name).strip()
    if _HAS_PANDAS and isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        col = y.columns[0]
        if col is not None and str(col).strip():
            return str(col).strip()
    return "target"


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _missing_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in ("f",):
        return np.isnan(values)
    if values.dtype.kind == "O":
        mask = np.zeros(len(values), dtype=bool)
        for i, v in enumerate(values):
            if v is None:
                mask[i] = True
            else:
                try:
                    if isinstance(v, float) and math.isnan(v):
                        mask[i] = True
                except Exception:
                    pass
        return mask
    return np.zeros(len(values), dtype=bool)


def _classify_values(
    values: np.ndarray,
    *,
    n_total: int,
    categorical_max_unique: int = 12,
    categorical_unique_ratio: float = 0.05,
) -> Tuple[str, bool, List[str]]:
    notes: List[str] = []

    finite = values[~_missing_mask(values)]
    if values.dtype.kind == "f":
        finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        notes.append("All values are missing or non-finite.")
        return T_UNKNOWN, False, notes

    unique = np.unique(finite)
    n_unique = unique.size

    if n_unique == 1:
        notes.append("Constant feature - no information.")
        return T_CONSTANT, False, notes

    if n_unique == 2:
        return T_BINARY, True, notes

    if values.dtype.kind in ("i", "u"):
        if n_unique <= categorical_max_unique or n_unique <= max(
            3, int(categorical_unique_ratio * max(n_total, 1))
        ):
            return T_CATEGORICAL, True, notes
        return T_CONTINUOUS, True, notes

    if values.dtype.kind == "f":
        is_integer_like = np.allclose(finite, np.round(finite), atol=1e-9)
        if is_integer_like and (
            n_unique <= categorical_max_unique
            or n_unique <= max(3, int(categorical_unique_ratio * max(n_total, 1)))
        ):
            notes.append("Integer-like float values; treated as categorical.")
            return T_CATEGORICAL, True, notes
        return T_CONTINUOUS, True, notes

    if n_unique <= categorical_max_unique:
        return T_CATEGORICAL, True, notes
    notes.append("Non-numeric values with high cardinality.")
    return T_CATEGORICAL, True, notes


def _correlation(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    target_type: str,
) -> Tuple[Optional[float], Optional[str]]:
    f_mask = ~_missing_mask(feature)
    t_mask = ~_missing_mask(target)
    mask = f_mask & t_mask
    if mask.sum() < 3:
        return None, None
    f = feature[mask]
    t = target[mask]
    try:
        f_num = f.astype(float)
        t_num = t.astype(float)
    except (TypeError, ValueError):
        return None, None
    if np.std(f_num) < 1e-12 or np.std(t_num) < 1e-12:
        return 0.0, "pearson"
    if target_type == T_CONTINUOUS:
        return float(np.corrcoef(f_num, t_num)[0, 1]), "pearson"
    if target_type == T_BINARY:
        return float(np.corrcoef(f_num, t_num)[0, 1]), "point_biserial"
    classes = np.unique(t_num)
    if classes.size < 2:
        return 0.0, "eta"
    grand_mean = f_num.mean()
    ss_between = 0.0
    ss_total = float(np.sum((f_num - grand_mean) ** 2))
    if ss_total < 1e-12:
        return 0.0, "eta"
    for c in classes:
        sub = f_num[t_num == c]
        if sub.size == 0:
            continue
        ss_between += sub.size * (sub.mean() - grand_mean) ** 2
    eta_sq = ss_between / ss_total
    eta = math.sqrt(max(0.0, min(1.0, eta_sq)))
    return float(eta), "eta"


# ---------------------------------------------------------------------------
# Validation and classification
# ---------------------------------------------------------------------------


def _validate_dataset(
    X: np.ndarray, y: Optional[np.ndarray], min_samples: int
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if X.ndim != 2:
        issues.append(f"Expected 2D feature matrix, got shape {X.shape}.")
    n_samples = X.shape[0] if X.ndim >= 1 else 0
    n_features = X.shape[1] if X.ndim == 2 else 0
    if n_samples == 0:
        issues.append("Dataset is empty (no samples).")
    if n_features == 0:
        issues.append("Dataset has no features.")
    if n_samples < min_samples:
        issues.append(
            f"Dataset has only {n_samples} samples, below the recommended "
            f"minimum of {min_samples}."
        )
    if y is not None:
        if y.shape[0] != n_samples:
            issues.append(
                f"Target length ({y.shape[0]}) does not match number of "
                f"samples ({n_samples})."
            )
    if X.size and X.dtype.kind == "f":
        non_finite = ~np.isfinite(X)
        if non_finite.any():
            n = int(non_finite.sum())
            issues.append(f"Feature matrix contains {n} non-finite values.")
    return len(issues) == 0, issues


def _summarise_target(y: np.ndarray) -> TargetSummary:
    missing = _missing_mask(y)
    n_missing = int(missing.sum())
    finite = y[~missing]
    if y.dtype.kind == "f":
        finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return TargetSummary(
            inferred_type=T_UNKNOWN, is_valid=False,
            n_unique=0, n_missing=n_missing,
            notes=["Target has no usable values."],
        )

    target_type, is_valid, notes = _classify_values(
        y, n_total=len(y),
    )
    summary = TargetSummary(
        inferred_type=target_type,
        is_valid=is_valid,
        n_unique=int(np.unique(finite).size),
        n_missing=n_missing,
        notes=list(notes),
    )

    try:
        finite_num = finite.astype(float)
        summary.min_val = _safe_float(np.min(finite_num))
        summary.max_val = _safe_float(np.max(finite_num))
        summary.mean = _safe_float(np.mean(finite_num))
        summary.std = _safe_float(np.std(finite_num))
        summary.median = _safe_float(np.median(finite_num))
    except (TypeError, ValueError):
        pass

    if target_type in (T_BINARY, T_CATEGORICAL):
        values, counts = np.unique(finite, return_counts=True)
        order = np.argsort(-counts)
        summary.class_counts = [
            (
                _safe_float(values[i]) if values.dtype.kind in ("i", "u", "f") else values[i],
                int(counts[i]),
            )
            for i in order
        ]
        if counts.size:
            summary.class_balance = float(counts.min() / counts.max())
            if summary.class_balance < 0.1:
                summary.notes.append(
                    f"Highly imbalanced classes (min/max ratio={summary.class_balance:.3f})."
                )

    return summary


def _summarise_feature(
    idx: int,
    name: str,
    column: np.ndarray,
    *,
    target: Optional[np.ndarray],
    target_type: Optional[str],
) -> FeatureSummary:
    missing = _missing_mask(column)
    n_missing = int(missing.sum())

    feature_type, is_valid, notes = _classify_values(
        column, n_total=len(column),
    )
    summary = FeatureSummary(
        index=idx,
        name=name,
        inferred_type=feature_type,
        is_valid=is_valid,
        n_unique=int(np.unique(column[~missing]).size) if (~missing).any() else 0,
        n_missing=n_missing,
        notes=list(notes),
    )
    if n_missing:
        summary.notes.append(f"{n_missing} missing value(s).")

    finite = column[~missing]
    try:
        finite_num = finite.astype(float)
        finite_num = finite_num[np.isfinite(finite_num)]
        if finite_num.size:
            summary.min_val = _safe_float(np.min(finite_num))
            summary.max_val = _safe_float(np.max(finite_num))
            summary.mean = _safe_float(np.mean(finite_num))
            summary.std = _safe_float(np.std(finite_num))
            summary.median = _safe_float(np.median(finite_num))
            summary.q1 = _safe_float(np.percentile(finite_num, 25))
            summary.q3 = _safe_float(np.percentile(finite_num, 75))
    except (TypeError, ValueError):
        pass

    if feature_type in (T_BINARY, T_CATEGORICAL) and finite.size:
        values, counts = np.unique(finite, return_counts=True)
        order = np.argsort(-counts)
        summary.top_categories = [
            (
                _safe_float(values[i]) if values.dtype.kind in ("i", "u", "f") else values[i],
                int(counts[i]),
            )
            for i in order[:10]
        ]

    if target is not None and target_type is not None and feature_type != T_CONSTANT:
        corr, kind = _correlation(column, target, target_type=target_type)
        summary.correlation_with_target = corr
        summary.correlation_kind = kind

    return summary


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _safe_filename(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:60] or "feature"


def _plot_continuous(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    sample_value: Optional[float] = None,
    exemplar_value: Optional[float] = None,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                color=COLORS["dirty_white"])
        ax.set_title(title)
        return _save(fig, out_path)

    n_bins = min(60, max(10, int(math.sqrt(finite.size))))
    ax.hist(
        finite, bins=n_bins, color=get_bar_color(0),
        edgecolor=COLORS["dirty_white"], alpha=0.85,
    )
    if sample_value is not None and np.isfinite(sample_value):
        ax.axvline(sample_value, color=_SAMPLE_COLOR, linewidth=3.0,
                   linestyle="--",
                   label=f"Sample ({sample_value:.3g})")
    if exemplar_value is not None and np.isfinite(exemplar_value):
        ax.axvline(exemplar_value, color=_EXEMPLAR_COLOR, linewidth=3.0,
                   linestyle=":",
                   label=f"Exemplar ({exemplar_value:.3g})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if sample_value is not None or exemplar_value is not None:
        legend = ax.legend(
            frameon=True, fancybox=True, facecolor=COLORS["dark_background"],
            edgecolor=COLORS["dirty_white"], fontsize=14,
        )
        for text in legend.get_texts():
            text.set_color(COLORS["dirty_white"])
    style_numerical_plot(ax)
    fig.tight_layout()
    return _save(fig, out_path)


def _plot_categorical(
    counts: List[Tuple[Any, int]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    sample_value: Optional[Any] = None,
    exemplar_value: Optional[Any] = None,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    if not counts:
        ax.text(0.5, 0.5, "No values", ha="center", va="center",
                color=COLORS["dirty_white"])
        ax.set_title(title)
        return _save(fig, out_path)

    labels = [str(v) for v, _ in counts]
    heights = [c for _, c in counts]
    colors = [get_bar_color(i) for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), heights, color=colors,
                  edgecolor=COLORS["dirty_white"], linewidth=2.0)
    for i, bar in enumerate(bars):
        bar.set_alpha(0.7 + 0.2 * (i % 2))

    def _match(val: Any) -> Optional[int]:
        if val is None:
            return None
        for i, (cat, _) in enumerate(counts):
            try:
                if cat is None:
                    continue
                if math.isclose(float(cat), float(val), abs_tol=1e-9):
                    return i
            except (TypeError, ValueError):
                if str(cat) == str(val):
                    return i
        return None

    sample_idx = _match(sample_value)
    if sample_idx is not None:
        bars[sample_idx].set_edgecolor(_SAMPLE_COLOR)
        bars[sample_idx].set_linewidth(4.0)
        ax.plot(
            sample_idx, heights[sample_idx] + max(heights) * 0.04, "v",
            color=_SAMPLE_COLOR, markersize=14,
            markeredgecolor=COLORS["dirty_white"], markeredgewidth=2,
            label=f"Sample ({sample_value})",
        )
    exemplar_idx = _match(exemplar_value)
    if exemplar_idx is not None:
        bars[exemplar_idx].set_edgecolor(_EXEMPLAR_COLOR)
        bars[exemplar_idx].set_linewidth(4.0)
        ax.plot(
            exemplar_idx, heights[exemplar_idx] + max(heights) * 0.10, "s",
            color=_EXEMPLAR_COLOR, markersize=14,
            markeredgecolor=COLORS["dirty_white"], markeredgewidth=2,
            label=f"Exemplar ({exemplar_value})",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if sample_idx is not None or exemplar_idx is not None:
        legend = ax.legend(
            frameon=True, fancybox=True, facecolor=COLORS["dark_background"],
            edgecolor=COLORS["dirty_white"], fontsize=14,
        )
        for text in legend.get_texts():
            text.set_color(COLORS["dirty_white"])
    style_categorical_plot(ax, num_categories=len(labels))
    fig.tight_layout()
    return _save(fig, out_path)


def _plot_correlation_heatmap(
    X_num: np.ndarray,
    feature_names: Sequence[str],
    out_path: Path,
) -> Optional[Path]:
    if X_num.shape[1] < 2:
        return None
    valid_cols = []
    valid_names = []
    for j in range(X_num.shape[1]):
        col = X_num[:, j]
        if np.isfinite(col).sum() >= 3 and np.nanstd(col) > 1e-12:
            valid_cols.append(col)
            valid_names.append(feature_names[j])
    if len(valid_cols) < 2:
        return None
    M = np.vstack(valid_cols)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(M)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    apply_style()
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(valid_names)),
                                    max(7, 0.45 * len(valid_names))))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(valid_names)))
    ax.set_yticks(range(len(valid_names)))
    ax.set_xticklabels(valid_names, rotation=45, ha="right",
                       color=COLORS["dirty_white"])
    ax.set_yticklabels(valid_names, color=COLORS["dirty_white"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=COLORS["dirty_white"])
    cbar.set_label("Pearson correlation", color=COLORS["dirty_white"])
    ax.set_title("Feature-Feature Correlation Matrix",
                 color=COLORS["dirty_white"], fontsize=22, pad=20)
    fig.patch.set_facecolor(COLORS["dark_background"])
    ax.set_facecolor(COLORS["plot_background"])
    fig.tight_layout()
    return _save(fig, out_path)


def _plot_target_correlation_bars(
    feature_summaries: List[FeatureSummary],
    out_path: Path,
) -> Optional[Path]:
    items = [
        (fs.name, fs.correlation_with_target)
        for fs in feature_summaries
        if fs.correlation_with_target is not None
    ]
    if not items:
        return None
    items.sort(key=lambda kv: -abs(kv[1]))
    names = [n for n, _ in items]
    values = [v for _, v in items]

    apply_style()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(names))))
    colors = [get_bar_color(0) if v >= 0 else get_bar_color(3) for v in values]
    ax.barh(range(len(names)), values, color=colors,
            edgecolor=COLORS["dirty_white"], linewidth=1.5, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=COLORS["dirty_white"])
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.axvline(0.0, color=COLORS["dirty_white"], linewidth=1.0)
    ax.set_xlabel("Correlation with target")
    ax.set_title("Feature - Target Association", pad=20)
    style_numerical_plot(ax)
    fig.tight_layout()
    return _save(fig, out_path)


def _find_category_index(categories: Sequence[Any], value: Any) -> Optional[int]:
    if value is None:
        return None
    for i, cat in enumerate(categories):
        try:
            if math.isclose(float(cat), float(value), abs_tol=1e-9):
                return i
        except (TypeError, ValueError):
            if str(cat) == str(value):
                return i
    return None


def _plot_feature_target_relationship(
    *,
    feature_values: np.ndarray,
    target_values: np.ndarray,
    feature_summary: FeatureSummary,
    target_type: str,
    target_name: str,
    out_path: Path,
    sample_feature_value: Optional[float],
    exemplar_feature_value: Optional[float],
    sample_target_value: Optional[float],
    exemplar_target_value: Optional[float],
) -> Optional[Path]:
    ftype = feature_summary.inferred_type
    if ftype in (T_CONSTANT, T_UNKNOWN) or target_type in (T_CONSTANT, T_UNKNOWN):
        return None

    fmask = ~_missing_mask(feature_values)
    tmask = ~_missing_mask(target_values)
    mask = fmask & tmask
    if mask.sum() < 3:
        return None

    f = feature_values[mask]
    t = target_values[mask]
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # continuous feature vs continuous target: 2D histogram
    if ftype == T_CONTINUOUS and target_type == T_CONTINUOUS:
        try:
            f_num = f.astype(float)
            t_num = t.astype(float)
        except (TypeError, ValueError):
            plt.close(fig)
            return None
        finite = np.isfinite(f_num) & np.isfinite(t_num)
        f_num = f_num[finite]
        t_num = t_num[finite]
        if f_num.size < 3:
            plt.close(fig)
            return None
        bins = min(60, max(10, int(math.sqrt(f_num.size))))
        h = ax.hist2d(f_num, t_num, bins=bins, cmap="viridis")
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=COLORS["dirty_white"])
        cbar.set_label("density", color=COLORS["dirty_white"])
        if sample_feature_value is not None and sample_target_value is not None:
            ax.scatter(
                [sample_feature_value], [sample_target_value], s=120, marker="v",
                color=_SAMPLE_COLOR, edgecolors=COLORS["dirty_white"], linewidths=1.5,
                label="Sample",
            )
        if exemplar_feature_value is not None and exemplar_target_value is not None:
            ax.scatter(
                [exemplar_feature_value], [exemplar_target_value], s=120, marker="s",
                color=_EXEMPLAR_COLOR, edgecolors=COLORS["dirty_white"], linewidths=1.5,
                label="Exemplar",
            )
        ax.set_xlabel(feature_summary.name)
        ax.set_ylabel(target_name)
        ax.set_title(f"{feature_summary.name} vs {target_name} (2D density)")
        if (sample_feature_value is not None and sample_target_value is not None) or (
            exemplar_feature_value is not None and exemplar_target_value is not None
        ):
            legend = ax.legend(frameon=True, fancybox=True,
                               facecolor=COLORS["dark_background"],
                               edgecolor=COLORS["dirty_white"], fontsize=14)
            for text in legend.get_texts():
                text.set_color(COLORS["dirty_white"])
        style_numerical_plot(ax)
        fig.tight_layout()
        return _save(fig, out_path)

    # continuous feature vs categorical/binary target
    if ftype == T_CONTINUOUS and target_type in (T_BINARY, T_CATEGORICAL):
        try:
            f_num = f.astype(float)
        except (TypeError, ValueError):
            plt.close(fig)
            return None
        classes = np.unique(t)
        bins = min(50, max(10, int(math.sqrt(f_num.size))))
        for i, cls in enumerate(classes):
            vals = f_num[t == cls]
            if vals.size == 0:
                continue
            ax.hist(
                vals, bins=bins, density=True, alpha=0.45,
                color=get_bar_color(i), edgecolor=COLORS["dirty_white"], linewidth=1.0,
                label=f"{target_name}={cls}",
            )
        if sample_feature_value is not None:
            ax.axvline(sample_feature_value, color=_SAMPLE_COLOR, linewidth=3.0,
                       linestyle="--", label="Sample feature value")
        if exemplar_feature_value is not None:
            ax.axvline(exemplar_feature_value, color=_EXEMPLAR_COLOR, linewidth=3.0,
                       linestyle=":", label="Exemplar feature value")
        ax.set_xlabel(feature_summary.name)
        ax.set_ylabel("density")
        ax.set_title(f"{feature_summary.name} distribution by {target_name} class")
        legend = ax.legend(frameon=True, fancybox=True, facecolor=COLORS["dark_background"],
                           edgecolor=COLORS["dirty_white"], fontsize=12)
        for text in legend.get_texts():
            text.set_color(COLORS["dirty_white"])
        style_numerical_plot(ax)
        fig.tight_layout()
        return _save(fig, out_path)

    # categorical/binary feature vs continuous target
    if ftype in (T_BINARY, T_CATEGORICAL) and target_type == T_CONTINUOUS:
        categories = np.unique(f)
        if categories.size == 0:
            plt.close(fig)
            return None
        grouped = []
        labels = []
        for cat in categories:
            vals = t[f == cat]
            try:
                vals = vals.astype(float)
            except (TypeError, ValueError):
                continue
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            grouped.append(vals)
            labels.append(str(cat))
        if not grouped:
            plt.close(fig)
            return None
        bp = ax.boxplot(grouped, patch_artist=True, labels=labels)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(get_bar_color(i))
            box.set_edgecolor(COLORS["dirty_white"])
            box.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(COLORS["dirty_white"])
            med.set_linewidth(2.0)
        sidx = _find_category_index(categories, sample_feature_value)
        if sidx is not None and sample_target_value is not None:
            ax.scatter(
                [sidx + 1], [sample_target_value], marker="v", s=120,
                color=_SAMPLE_COLOR, edgecolors=COLORS["dirty_white"], linewidths=1.5,
                label="Sample",
            )
        eidx = _find_category_index(categories, exemplar_feature_value)
        if eidx is not None and exemplar_target_value is not None:
            ax.scatter(
                [eidx + 1], [exemplar_target_value], marker="s", s=120,
                color=_EXEMPLAR_COLOR, edgecolors=COLORS["dirty_white"], linewidths=1.5,
                label="Exemplar",
            )
        ax.set_xlabel(feature_summary.name)
        ax.set_ylabel(target_name)
        ax.set_title(f"{target_name} distribution by {feature_summary.name}")
        if (sidx is not None and sample_target_value is not None) or (
            eidx is not None and exemplar_target_value is not None
        ):
            legend = ax.legend(frameon=True, fancybox=True, facecolor=COLORS["dark_background"],
                               edgecolor=COLORS["dirty_white"], fontsize=12)
            for text in legend.get_texts():
                text.set_color(COLORS["dirty_white"])
        style_numerical_plot(ax)
        fig.tight_layout()
        return _save(fig, out_path)

    # categorical/binary feature vs categorical/binary target
    if ftype in (T_BINARY, T_CATEGORICAL) and target_type in (T_BINARY, T_CATEGORICAL):
        f_cats = np.unique(f)
        t_cats = np.unique(t)
        if f_cats.size == 0 or t_cats.size == 0:
            plt.close(fig)
            return None
        counts = np.zeros((len(f_cats), len(t_cats)), dtype=float)
        for i, fc in enumerate(f_cats):
            for j, tc in enumerate(t_cats):
                counts[i, j] = np.sum((f == fc) & (t == tc))
        im = ax.imshow(counts, cmap="viridis", aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=COLORS["dirty_white"])
        cbar.set_label("count", color=COLORS["dirty_white"])
        ax.set_xticks(range(len(t_cats)))
        ax.set_yticks(range(len(f_cats)))
        ax.set_xticklabels([str(v) for v in t_cats], rotation=45, ha="right",
                           color=COLORS["dirty_white"])
        ax.set_yticklabels([str(v) for v in f_cats], color=COLORS["dirty_white"])
        for i in range(len(f_cats)):
            for j in range(len(t_cats)):
                ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                        color=COLORS["dirty_white"], fontsize=10)
        sfx = _find_category_index(f_cats, sample_feature_value)
        sty = _find_category_index(t_cats, sample_target_value)
        if sfx is not None and sty is not None:
            ax.scatter(sty, sfx, marker="v", s=150, color=_SAMPLE_COLOR,
                       edgecolors=COLORS["dirty_white"], linewidths=1.5, label="Sample")
        efx = _find_category_index(f_cats, exemplar_feature_value)
        ety = _find_category_index(t_cats, exemplar_target_value)
        if efx is not None and ety is not None:
            ax.scatter(ety, efx, marker="s", s=150, color=_EXEMPLAR_COLOR,
                       edgecolors=COLORS["dirty_white"], linewidths=1.5, label="Exemplar")
        ax.set_xlabel(target_name)
        ax.set_ylabel(feature_summary.name)
        ax.set_title(f"{feature_summary.name} vs {target_name} contingency")
        if (sfx is not None and sty is not None) or (efx is not None and ety is not None):
            legend = ax.legend(frameon=True, fancybox=True, facecolor=COLORS["dark_background"],
                               edgecolor=COLORS["dirty_white"], fontsize=12)
            for text in legend.get_texts():
                text.set_color(COLORS["dirty_white"])
        style_numerical_plot(ax)
        fig.tight_layout()
        return _save(fig, out_path)

    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def _format_target(t: TargetSummary) -> List[str]:
    lines = []
    lines.append(f"  inferred_type     : {t.inferred_type}")
    lines.append(f"  is_valid          : {t.is_valid}")
    lines.append(f"  n_unique          : {t.n_unique}")
    lines.append(f"  n_missing         : {t.n_missing}")
    if t.min_val is not None:
        lines.append(f"  min / max         : {t.min_val:.6g} / {t.max_val:.6g}")
    if t.mean is not None:
        lines.append(f"  mean / std        : {t.mean:.6g} / {t.std:.6g}")
    if t.median is not None:
        lines.append(f"  median            : {t.median:.6g}")
    if t.class_counts:
        lines.append("  class_counts      :")
        for cls, count in t.class_counts:
            lines.append(f"      {cls!s:<20s} -> {count}")
        if t.class_balance is not None:
            lines.append(f"  class_balance     : {t.class_balance:.3f} (min/max)")
    if t.notes:
        lines.append("  notes             :")
        for n in t.notes:
            lines.append(f"      - {n}")
    return lines


def _format_feature(f: FeatureSummary) -> List[str]:
    lines = []
    lines.append(f"  [{f.index:3d}] {f.name}")
    lines.append(f"        type           : {f.inferred_type}")
    lines.append(f"        is_valid       : {f.is_valid}")
    lines.append(f"        n_unique       : {f.n_unique}")
    lines.append(f"        n_missing      : {f.n_missing}")
    if f.min_val is not None:
        lines.append(f"        min / max      : {f.min_val:.6g} / {f.max_val:.6g}")
    if f.mean is not None:
        lines.append(
            f"        mean / std     : {f.mean:.6g} / {f.std:.6g}"
        )
    if f.median is not None and f.q1 is not None and f.q3 is not None:
        lines.append(
            f"        q1 / med / q3  : {f.q1:.6g} / {f.median:.6g} / {f.q3:.6g}"
        )
    if f.top_categories:
        cats = ", ".join(f"{c!s}:{n}" for c, n in f.top_categories[:6])
        lines.append(f"        top_categories : {cats}")
    if f.correlation_with_target is not None:
        kind = f.correlation_kind or "corr"
        lines.append(
            f"        corr_w/target  : {f.correlation_with_target:+.4f} ({kind})"
        )
    if f.notes:
        for n in f.notes:
            lines.append(f"        note           : {n}")
    return lines


def _write_summary_text(
    report: DatasetReport,
    *,
    sample: Optional[np.ndarray],
    exemplar: Optional[np.ndarray],
    output_path: Path,
) -> Path:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"DATASET ANALYSIS SUMMARY: {report.dataset_key}")
    lines.append("=" * 78)
    lines.append(f"n_samples         : {report.n_samples}")
    lines.append(f"n_features        : {report.n_features}")
    lines.append(f"target_name       : {report.target_name}")
    lines.append(f"is_valid          : {report.is_valid}")
    if report.issues:
        lines.append("issues:")
        for issue in report.issues:
            lines.append(f"  - {issue}")

    lines.append("")
    lines.append("-" * 78)
    lines.append("TARGET")
    lines.append("-" * 78)
    if report.target is not None:
        lines.extend(_format_target(report.target))
    else:
        lines.append("  (no target supplied)")

    lines.append("")
    lines.append("-" * 78)
    lines.append("FEATURES")
    lines.append("-" * 78)
    for f in report.features:
        lines.extend(_format_feature(f))
        lines.append("")

    if sample is not None or exemplar is not None:
        lines.append("-" * 78)
        lines.append("SAMPLE / EXEMPLAR REFERENCE VALUES")
        lines.append("-" * 78)
        for f in report.features:
            row = f"  [{f.index:3d}] {f.name:30s}"
            if sample is not None and f.index < len(sample):
                row += f"  sample={float(sample[f.index]):+12.6g}"
            if exemplar is not None and f.index < len(exemplar):
                row += f"  exemplar={float(exemplar[f.index]):+12.6g}"
            lines.append(row)

    if report.saved_plots:
        lines.append("")
        lines.append("-" * 78)
        lines.append("PLOTS")
        lines.append("-" * 78)
        for p in report.saved_plots:
            lines.append(f"  {p}")

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_dataset(
    X: Any,
    y: Optional[Any] = None,
    *,
    feature_names: Optional[Sequence[str]] = None,
    target_name: Optional[str] = None,
    dataset_key: str = "dataset",
    output_dir: Union[str, Path] = "dataset_analysis",
    sample: Optional[Sequence[float]] = None,
    exemplar: Optional[Sequence[float]] = None,
    sample_target_value: Optional[float] = None,
    exemplar_target_value: Optional[float] = None,
    min_samples: int = 10,
    plot_features: bool = True,
    plot_correlation: bool = True,
    write_summary: bool = True,
) -> DatasetReport:
    """Run a multi-stage diagnostic on the dataset.

    Parameters
    ----------
    X : array-like or pandas DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : array-like, optional
        Target vector of length n_samples. If omitted, only feature-side
        diagnostics are produced.
    feature_names : sequence of str, optional
        Used in plots/text. Falls back to ``X.columns`` for DataFrames or
        ``feature_<i>`` otherwise.
    dataset_key : str
        Short identifier used in plot titles and the summary header.
    target_name : str, optional
        Display name of the target shown in plots and summary text. If not
        provided, inferred from ``y`` when possible (e.g., pandas Series
        name) and otherwise defaults to ``\"target\"``.
    output_dir : str or Path
        Folder in which plots and summary text are written.
    sample, exemplar : array-like, optional
        Reference rows of length n_features. When provided, their values are
        overlaid on the per-feature distribution plots.
    sample_target_value, exemplar_target_value : float, optional
        Optional target values corresponding to ``sample`` and ``exemplar``.
        When provided, they are overlaid on the target plot and feature-target
        relationship plots.
    plot_features : bool
        If False, skip per-feature distribution plots.
    plot_correlation : bool
        If False, skip the correlation heatmap and target-correlation bars.
    write_summary : bool
        If False, skip writing the textual summary.

    Returns
    -------
    DatasetReport
        Structured report with feature/target summaries and the list of
        files produced.
    """

    X_arr, derived_names = _to_2d_array(X)
    y_arr: Optional[np.ndarray] = None
    if y is not None:
        y_arr = _to_1d_array(y)
    resolved_target_name = _infer_target_name(y, target_name)

    if feature_names is None:
        feature_names = derived_names
    feature_names = [str(n) for n in feature_names]
    if len(feature_names) != X_arr.shape[1]:
        feature_names = derived_names

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    is_valid, issues = _validate_dataset(X_arr, y_arr, min_samples=min_samples)
    if not is_valid:
        for issue in issues:
            logger.warning("Dataset validation: %s", issue)

    target_summary: Optional[TargetSummary] = None
    target_type: Optional[str] = None
    saved_plots: List[Path] = []

    if y_arr is not None and y_arr.size:
        target_summary = _summarise_target(y_arr)
        target_type = target_summary.inferred_type
        try:
            if target_type == T_CONTINUOUS:
                p = _plot_continuous(
                    y_arr.astype(float),
                    title=f"{resolved_target_name} distribution ({dataset_key})",
                    xlabel=resolved_target_name,
                    ylabel="count",
                    out_path=out / "target_distribution.png",
                    sample_value=sample_target_value,
                    exemplar_value=exemplar_target_value,
                )
                saved_plots.append(p)
            elif target_summary.class_counts:
                p = _plot_categorical(
                    target_summary.class_counts,
                    title=f"{resolved_target_name} class counts ({dataset_key})",
                    xlabel="class",
                    ylabel="count",
                    out_path=out / "target_distribution.png",
                    sample_value=sample_target_value,
                    exemplar_value=exemplar_target_value,
                )
                saved_plots.append(p)
        except Exception as exc:
            logger.warning("Failed to plot target distribution: %s", exc)

    sample_arr: Optional[np.ndarray] = None
    exemplar_arr: Optional[np.ndarray] = None
    if sample is not None:
        sample_arr = np.asarray(sample, dtype=float).flatten()
    if exemplar is not None:
        exemplar_arr = np.asarray(exemplar, dtype=float).flatten()

    feature_summaries: List[FeatureSummary] = []
    feature_dir = out / "features"
    feature_target_dir = out / "features_target"
    if plot_features:
        feature_dir.mkdir(exist_ok=True)
        if y_arr is not None:
            feature_target_dir.mkdir(exist_ok=True)

    for j in range(X_arr.shape[1]):
        col = X_arr[:, j]
        fs = _summarise_feature(
            j, feature_names[j], col,
            target=y_arr, target_type=target_type,
        )
        feature_summaries.append(fs)

        if not plot_features:
            continue
        try:
            sval = (float(sample_arr[j])
                    if sample_arr is not None and j < sample_arr.size
                    else None)
            eval_ = (float(exemplar_arr[j])
                     if exemplar_arr is not None and j < exemplar_arr.size
                     else None)
            base = f"{fs.index:03d}_{_safe_filename(fs.name)}"
            if fs.inferred_type == T_CONTINUOUS:
                col_finite = col[np.isfinite(col.astype(float, copy=False))]
                p = _plot_continuous(
                    col_finite.astype(float),
                    title=f"Feature [{fs.index}] {fs.name}",
                    xlabel=fs.name,
                    ylabel="count",
                    out_path=feature_dir / f"{base}.png",
                    sample_value=sval,
                    exemplar_value=eval_,
                )
                saved_plots.append(p)
            elif fs.inferred_type in (T_BINARY, T_CATEGORICAL) and fs.top_categories:
                p = _plot_categorical(
                    fs.top_categories,
                    title=f"Feature [{fs.index}] {fs.name}",
                    xlabel="value",
                    ylabel="count",
                    out_path=feature_dir / f"{base}.png",
                    sample_value=sval,
                    exemplar_value=eval_,
                )
                saved_plots.append(p)

            if y_arr is not None and target_type is not None:
                p_rel = _plot_feature_target_relationship(
                    feature_values=col,
                    target_values=y_arr,
                    feature_summary=fs,
                    target_type=target_type,
                    target_name=resolved_target_name,
                    out_path=feature_target_dir / f"{base}.png",
                    sample_feature_value=sval,
                    exemplar_feature_value=eval_,
                    sample_target_value=sample_target_value,
                    exemplar_target_value=exemplar_target_value,
                )
                if p_rel is not None:
                    saved_plots.append(p_rel)
        except Exception as exc:
            logger.warning("Failed to plot feature %s: %s", fs.name, exc)

    if plot_correlation:
        try:
            X_float = X_arr.astype(float, copy=False)
            heat = _plot_correlation_heatmap(
                X_float,
                feature_names,
                out_path=out / "correlation_matrix.png",
            )
            if heat is not None:
                saved_plots.append(heat)
        except Exception as exc:
            logger.warning("Failed to plot correlation heatmap: %s", exc)

        if y_arr is not None:
            try:
                bars = _plot_target_correlation_bars(
                    feature_summaries,
                    out_path=out / "feature_target_correlation.png",
                )
                if bars is not None:
                    saved_plots.append(bars)
            except Exception as exc:
                logger.warning("Failed to plot feature-target correlation: %s", exc)

    report = DatasetReport(
        dataset_key=dataset_key,
        n_samples=int(X_arr.shape[0]) if X_arr.ndim == 2 else 0,
        n_features=int(X_arr.shape[1]) if X_arr.ndim == 2 else 0,
        feature_names=list(feature_names),
        target_name=resolved_target_name,
        is_valid=is_valid,
        issues=issues,
        target=target_summary,
        features=feature_summaries,
        saved_plots=saved_plots,
    )

    if write_summary:
        try:
            text_path = _write_summary_text(
                report,
                sample=sample_arr,
                exemplar=exemplar_arr,
                output_path=out / "summary.txt",
            )
            report.summary_text_path = text_path
        except Exception as exc:
            logger.warning("Failed to write summary text: %s", exc)

    logger.info(
        "Dataset analysis for '%s' complete: %d feature(s) inspected, "
        "%d plot(s) saved in %s",
        dataset_key, len(feature_summaries), len(saved_plots), out,
    )
    if report.summary_text_path is not None:
        logger.info("Wrote summary text to %s", report.summary_text_path)

    return report


__all__ = [
    "analyze_dataset",
    "DatasetReport",
    "FeatureSummary",
    "TargetSummary",
    "T_BINARY",
    "T_CATEGORICAL",
    "T_CONTINUOUS",
    "T_CONSTANT",
    "T_UNKNOWN",
]
