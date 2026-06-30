"""Priorities analyser.

Given a model, a dataset, one or more target prediction values, and a
priorities dict (the one consumed by ``MINLSearchExplainer``), this module
produces a structured diagnostic that helps a user reason about *whether
the configured priorities cover a reasonable portion of the dataset* and
*how the resulting search space relates to interesting exemplars*.

The diagnostic has four pieces:

1. Per-feature **coverage** -- for each numerical feature it reports the
   priority bounds, the dataset min/max, the percentage of the dataset
   range that the bounds cover, and the percentage of dataset rows whose
   value falls inside the allowed (positive-priority) intervals. The
   format mirrors ``MINLSearchExplainer._log_workflow_initial_and_bounds``
   so callers can compare numbers across surfaces.

2. **Closest exemplars** -- the dataset rows whose model prediction is
   nearest to each requested target value.

3. **Dataset distribution plots** -- delegates to
   :func:`explainit.utils.dataset_analyzer.analyze_dataset` so the user
   gets feature distributions, feature/target relationship plots and
   correlation matrices. Sample/exemplar pairs are overlaid.

4. **Priority plots** -- delegates to
   :func:`explainit.utils.priority_plots.plot_priorities` to render the
   numerical/categorical priority surfaces. All sample/exemplar pairs are
   shown on every plot, linked by translucent connectors.

Typical usage::

    from explainit.utils.priorities_analyser import analyse_priorities

    report = analyse_priorities(
        model=ctx.model,
        dataset=ctx.X_train,
        target_values=[0.25, 0.75],
        priorities=priorities,
        feature_names=ctx.feature_names,
        samples=[sample_a, sample_b],     # optional
        y_full=ctx.y_train,
        target_name="disease_progression",
        dataset_key="diabetes",
        output_dir=Path("analysis/diabetes/multi"),
    )

When ``exemplars`` is omitted, one exemplar is computed per ``target_y``
by picking the dataset row with prediction closest to that target. When
``samples`` is omitted the priority/dataset plots simply show exemplars
(or nothing, when both are missing).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from explainit.utils.dataset_analyzer import analyze_dataset, DatasetReport
from explainit.utils.priority_plots import plot_priorities


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureCoverage:
    """Per-feature coverage row matching the MINLP workflow log layout."""

    index: int
    name: str
    bounds_min: float
    bounds_max: float
    dataset_min: float
    dataset_max: float
    allowed_space_pct: float
    allowed_points_pct: float
    allowed_intervals: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ExemplarInfo:
    """An exemplar row picked from the dataset."""

    target_value: float
    row_index: int
    vector: np.ndarray
    prediction: float


@dataclass
class PriorityAnalysisReport:
    dataset_key: str
    coverages: List[FeatureCoverage]
    global_allowed_pct: float
    n_rows_in_dataset: int
    exemplars: List[ExemplarInfo]
    saved_plots: List[Path] = field(default_factory=list)
    coverage_text_path: Optional[Path] = None
    summary_text_path: Optional[Path] = None
    dataset_report: Optional[DatasetReport] = None


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Internal helpers (mostly mirroring MINLSearchExplainer geometry)
# ---------------------------------------------------------------------------


def _to_numpy(X: Any) -> np.ndarray:
    if hasattr(X, "values"):
        return np.asarray(X.values, dtype=float)
    return np.asarray(X, dtype=float)


def _in_intervals(value: float, intervals: Sequence[Tuple[float, float]], tol: float = 1e-12) -> bool:
    for lo, hi in intervals:
        if (value >= lo - tol) and (value <= hi + tol):
            return True
    return False


def _extract_positive_intervals(
    x_grid: np.ndarray, y_grid: np.ndarray, positive_eps: float = 1e-12,
) -> List[Tuple[float, float]]:
    mask = np.asarray(y_grid, dtype=float) > float(positive_eps)
    intervals: List[Tuple[float, float]] = []
    i = 0
    n = len(x_grid)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        intervals.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return intervals


def _derive_intervals_from_fn(
    fn: Callable[[float], float],
    dataset_min: float,
    dataset_max: float,
    grid_size: int = 2000,
) -> List[Tuple[float, float]]:
    if abs(dataset_max - dataset_min) < 1e-12:
        return [(dataset_min, dataset_max)]
    x_grid = np.linspace(dataset_min, dataset_max, int(grid_size))
    y_values: List[float] = []
    for x in x_grid:
        try:
            y = float(fn(float(x)))
        except Exception:
            y = 0.0
        if not math.isfinite(y):
            y = 0.0
        y_values.append(y)
    y_grid = np.asarray(y_values, dtype=float)
    return _extract_positive_intervals(x_grid, y_grid)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def compute_priority_coverage(
    priorities: Dict[str, Any],
    dataset: Any,
    feature_names: Optional[Sequence[str]] = None,
    *,
    grid_size: int = 2000,
) -> Tuple[List[FeatureCoverage], float, int]:
    """Compute per-feature coverage statistics.

    Returns ``(per_feature_coverages, global_allowed_pct, n_rows)``. The
    bounds reported are derived from the priority dict's ``min/max`` (or,
    when missing/None, the dataset min/max), and the allowed intervals are
    inferred from the priority function values (same logic
    ``MINLSearchExplainer`` uses internally before the search).
    """

    X = _to_numpy(dataset)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D dataset, got shape {X.shape}.")
    n_rows = X.shape[0]
    numerical: Dict[int, Any] = priorities.get("numerical", {}) or {}

    coverages: List[FeatureCoverage] = []
    global_mask = np.ones(n_rows, dtype=bool)

    for idx in sorted(numerical.keys()):
        cfg = numerical[idx]
        if not isinstance(cfg, dict):
            continue
        col = X[:, idx]
        dmin = float(np.min(col))
        dmax = float(np.max(col))
        fn = cfg.get("function")
        explicit_min = cfg.get("min")
        explicit_max = cfg.get("max")

        if fn is None:
            fixed = (
                float(explicit_min)
                if explicit_min is not None
                else float(explicit_max if explicit_max is not None else dmin)
            )
            intervals: List[Tuple[float, float]] = [(fixed, fixed)]
            bounds_min = fixed
            bounds_max = fixed
        else:
            existing = cfg.get("allowed_intervals")
            if existing:
                intervals = [(float(lo), float(hi)) for lo, hi in existing]
            else:
                intervals = _derive_intervals_from_fn(fn, dmin, dmax, grid_size=grid_size)
            if intervals:
                derived_min = float(intervals[0][0])
                derived_max = float(intervals[-1][1])
            else:
                derived_min = dmin
                derived_max = dmax
            bounds_min = (
                float(explicit_min)
                if explicit_min is not None and float(explicit_min) > derived_min
                else derived_min
            )
            bounds_max = (
                float(explicit_max)
                if explicit_max is not None and float(explicit_max) < derived_max
                else derived_max
            )

        dataset_span = dmax - dmin
        allowed_span = bounds_max - bounds_min
        if abs(dataset_span) < 1e-12:
            allowed_space_pct = 100.0
        else:
            allowed_space_pct = (allowed_span / dataset_span) * 100.0

        if intervals:
            row_mask = np.array(
                [_in_intervals(float(v), intervals) for v in col],
                dtype=bool,
            )
        else:
            row_mask = (col >= bounds_min) & (col <= bounds_max)
        allowed_points_pct = (
            100.0 * float(np.sum(row_mask)) / float(n_rows) if n_rows else 0.0
        )

        global_mask = np.logical_and(global_mask, row_mask)

        name = (
            feature_names[idx]
            if feature_names is not None and idx < len(feature_names)
            else f"feature_{idx}"
        )
        coverages.append(
            FeatureCoverage(
                index=idx,
                name=name,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                dataset_min=dmin,
                dataset_max=dmax,
                allowed_space_pct=float(allowed_space_pct),
                allowed_points_pct=float(allowed_points_pct),
                allowed_intervals=intervals,
            )
        )

    global_pct = 100.0 * float(np.sum(global_mask)) / float(n_rows) if n_rows else 0.0
    return coverages, float(global_pct), n_rows


def format_coverage_report(
    coverages: Sequence[FeatureCoverage],
    global_allowed_pct: float,
    n_rows: int,
    *,
    header: str = "Bounds per feature vs dataset min/max",
) -> str:
    """Render the coverage block in the same layout used by the MINLP workflow log."""

    lines: List[str] = []
    lines.append(f"{header}:")
    for cov in coverages:
        lines.append(
            f"   - {cov.name} (idx={cov.index}): "
            f"bounds=[{cov.bounds_min:.6f}, {cov.bounds_max:.6f}] | "
            f"dataset=[{cov.dataset_min:.6f}, {cov.dataset_max:.6f}] | "
            f"allowed space={cov.allowed_space_pct:.2f}% | "
            f"allowed points={cov.allowed_points_pct:.2f}%"
        )
    lines.append("")
    lines.append(
        "Global allowed samples (all numerical-feature bounds simultaneously): "
        f"{global_allowed_pct:.2f}% "
        f"({int(round(global_allowed_pct * n_rows / 100.0))}/{n_rows})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exemplars
# ---------------------------------------------------------------------------


def _coerce_model_pred(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    if callable(model) and not hasattr(model, "predict"):
        def _call(X):
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return np.asarray(model(arr)).flatten()
        return _call

    def _call(X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = model.predict(arr, verbose=0)
        return np.asarray(out).flatten()
    return _call


def find_closest_exemplars(
    model: Any,
    dataset: Any,
    target_y: float,
    *,
    k: int = 1,
) -> List[ExemplarInfo]:
    """Return the ``k`` dataset rows whose model prediction is closest to ``target_y``.

    The returned list is ordered by ascending distance to ``target_y``.
    """

    X = _to_numpy(dataset)
    if X.ndim != 2 or X.size == 0:
        return []
    predict = _coerce_model_pred(model)
    preds = predict(X)
    order = np.argsort(np.abs(preds - float(target_y)))
    take = min(int(k), order.size)
    out: List[ExemplarInfo] = []
    for i in order[:take]:
        row = X[int(i)].astype(float)
        out.append(
            ExemplarInfo(
                target_value=float(target_y),
                row_index=int(i),
                vector=row,
                prediction=float(preds[int(i)]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _format_target_for_filename(value: float) -> str:
    s = f"{float(value):g}"
    return s.replace("-", "neg").replace(".", "p")


def analyse_priorities(
    *,
    model: Any,
    dataset: Any,
    priorities: Dict[str, Any],
    feature_names: Optional[Sequence[str]] = None,
    target_values: Optional[Union[float, Sequence[float]]] = None,
    samples: Optional[Sequence[Sequence[float]]] = None,
    exemplars: Optional[Sequence[Sequence[float]]] = None,
    y_full: Optional[Sequence[float]] = None,
    target_name: str = "target",
    dataset_key: str = "dataset",
    output_dir: PathLike = "priorities_analysis",
    write_text: bool = True,
    run_dataset_analysis: bool = True,
    run_priority_plots: bool = True,
) -> PriorityAnalysisReport:
    """Run the four diagnostics (coverage, exemplars, dataset plots, priority plots).

    Parameters
    ----------
    model
        Keras model or a callable ``X -> prediction`` used to score exemplars
        and to evaluate model predictions on samples for overlay plots.
    dataset
        Feature matrix (e.g. training split) used for coverage statistics
        and as the candidate pool for closest exemplars. ``pandas`` and
        ``numpy`` inputs are both accepted.
    priorities
        The priorities dict consumed by ``MINLSearchExplainer``.
    feature_names
        Optional feature names for the coverage report and plots.
    target_values
        Either a single scalar or a sequence of scalars. One exemplar per
        target is derived if ``exemplars`` is not supplied.
    samples
        Optional list of sample vectors (one per "what-if" point of
        interest). All of them are overlaid on every plot.
    exemplars
        Optional list of exemplar vectors. When omitted, one exemplar is
        computed per ``target_values`` entry.
    y_full
        Optional ground-truth target vector passed through to the dataset
        analyzer. Has no effect on coverage.
    output_dir
        Where to write plots and the textual reports.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_names_list: Optional[List[str]] = (
        [str(n) for n in feature_names] if feature_names is not None else None
    )

    coverages, global_pct, n_rows = compute_priority_coverage(
        priorities, dataset, feature_names=feature_names_list,
    )
    coverage_text = format_coverage_report(coverages, global_pct, n_rows)
    for line in coverage_text.splitlines():
        logger.info(line)
    coverage_path: Optional[Path] = None
    if write_text:
        coverage_path = _write_text(out / "coverage.txt", coverage_text + "\n")

    target_list: List[float] = []
    if target_values is not None:
        if isinstance(target_values, (int, float, np.integer, np.floating)):
            target_list = [float(target_values)]
        else:
            target_list = [float(v) for v in target_values]

    exemplar_list: List[ExemplarInfo] = []
    if exemplars is not None:
        predict = _coerce_model_pred(model)
        for i, vec in enumerate(exemplars):
            arr = np.asarray(vec, dtype=float).flatten()
            target_v = (
                float(target_list[i])
                if i < len(target_list)
                else float(predict(arr.reshape(1, -1))[0])
            )
            pred = float(predict(arr.reshape(1, -1))[0])
            exemplar_list.append(
                ExemplarInfo(
                    target_value=target_v,
                    row_index=-1,
                    vector=arr,
                    prediction=pred,
                )
            )
    else:
        for ty in target_list:
            picked = find_closest_exemplars(model, dataset, ty, k=1)
            if picked:
                exemplar_list.append(picked[0])

    for ex in exemplar_list:
        logger.info(
            "Closest exemplar for target=%.4f: row=%d pred=%.4f",
            ex.target_value, ex.row_index, ex.prediction,
        )

    sample_vectors: Optional[List[np.ndarray]] = None
    if samples is not None:
        sample_vectors = [
            np.asarray(s, dtype=float).flatten() for s in samples
        ]

    exemplar_vectors: List[np.ndarray] = [ex.vector for ex in exemplar_list]

    n_pairs = max(
        len(sample_vectors) if sample_vectors is not None else 0,
        len(exemplar_vectors),
        0,
    )

    sample_target_values: List[Optional[float]] = []
    exemplar_target_values: List[Optional[float]] = []
    if n_pairs > 0:
        predict = _coerce_model_pred(model)
        for i in range(n_pairs):
            if sample_vectors is not None and i < len(sample_vectors):
                try:
                    sample_target_values.append(
                        float(predict(sample_vectors[i].reshape(1, -1))[0])
                    )
                except Exception as exc:
                    logger.warning("Sample prediction failed: %s", exc)
                    sample_target_values.append(None)
            else:
                sample_target_values.append(None)
            if i < len(exemplar_list):
                exemplar_target_values.append(exemplar_list[i].prediction)
            else:
                exemplar_target_values.append(None)

    saved_plots: List[Path] = []
    dataset_report: Optional[DatasetReport] = None

    if run_dataset_analysis:
        try:
            dataset_dir = out / "dataset"
            dataset_report = analyze_dataset(
                X=dataset,
                y=np.asarray(y_full).flatten() if y_full is not None else None,
                feature_names=feature_names_list,
                dataset_key=dataset_key,
                target_name=target_name,
                output_dir=dataset_dir,
                sample=sample_vectors,
                exemplar=exemplar_vectors if exemplar_vectors else None,
                sample_target_value=sample_target_values or None,
                exemplar_target_value=exemplar_target_values or None,
            )
            saved_plots.extend(dataset_report.saved_plots)
        except Exception as exc:
            logger.warning("Dataset analysis failed: %s", exc)

    if run_priority_plots:
        try:
            priority_dir = out / "priorities"
            priority_dir.mkdir(parents=True, exist_ok=True)
            target_threshold = target_list[0] if len(target_list) == 1 else None
            feature_matrix = _to_numpy(dataset)
            target_arr = (
                np.asarray(y_full, dtype=float).flatten()
                if y_full is not None else None
            )
            written = plot_priorities(
                priorities,
                sample=sample_vectors,
                exemplar=exemplar_vectors if exemplar_vectors else None,
                sample_target_value=sample_target_values or None,
                exemplar_target_value=exemplar_target_values or None,
                feature_matrix=feature_matrix,
                target_values=target_arr,
                target_name=target_name,
                target_threshold=target_threshold,
                feature_names=feature_names_list,
                save_dir=priority_dir,
                show=False,
            )
            saved_plots.extend(written)
        except Exception as exc:
            logger.warning("Priority plots failed: %s", exc)

    report = PriorityAnalysisReport(
        dataset_key=dataset_key,
        coverages=coverages,
        global_allowed_pct=float(global_pct),
        n_rows_in_dataset=int(n_rows),
        exemplars=exemplar_list,
        saved_plots=saved_plots,
        coverage_text_path=coverage_path,
        dataset_report=dataset_report,
    )

    if write_text:
        try:
            report.summary_text_path = _write_text(
                out / "analysis_summary.txt",
                _format_summary(report, samples=sample_vectors, target_name=target_name),
            )
        except Exception as exc:
            logger.warning("Failed to write analysis summary: %s", exc)

    return report


def _format_summary(
    report: PriorityAnalysisReport,
    *,
    samples: Optional[Sequence[np.ndarray]],
    target_name: str,
) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"PRIORITY ANALYSIS SUMMARY: {report.dataset_key}")
    lines.append("=" * 78)
    lines.append(format_coverage_report(
        report.coverages, report.global_allowed_pct, report.n_rows_in_dataset,
    ))
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"CLOSEST EXEMPLARS (target={target_name})")
    lines.append("-" * 78)
    if report.exemplars:
        for i, ex in enumerate(report.exemplars):
            lines.append(
                f"  #{i + 1}  target={ex.target_value:+.6f}  "
                f"row={ex.row_index:>6d}  pred={ex.prediction:+.6f}"
            )
    else:
        lines.append("  (no exemplars computed)")

    if samples:
        lines.append("")
        lines.append("-" * 78)
        lines.append("SAMPLES")
        lines.append("-" * 78)
        for i, s in enumerate(samples):
            preview = ", ".join(f"{float(v):+.3f}" for v in s[:6])
            more = "" if s.size <= 6 else f"  (+{s.size - 6} more)"
            lines.append(f"  #{i + 1}  [{preview}]{more}")

    if report.saved_plots:
        lines.append("")
        lines.append("-" * 78)
        lines.append("PLOTS")
        lines.append("-" * 78)
        for p in report.saved_plots:
            lines.append(f"  {p}")

    return "\n".join(lines) + "\n"


__all__ = [
    "FeatureCoverage",
    "ExemplarInfo",
    "PriorityAnalysisReport",
    "compute_priority_coverage",
    "format_coverage_report",
    "find_closest_exemplars",
    "analyse_priorities",
]
