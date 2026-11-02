"""``plot_priorities``: one priority plot per mutable feature."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from explainit.plotting._common import (
    PathLike,
    ensure_dir,
    group_path,
    is_actionable_numeric,
    numerical_path,
)
from explainit.plotting.categorical import plot_categorical_priority
from explainit.plotting.numerical import plot_numerical_priority


def plot_priorities(
    priorities: Mapping[str, Any],
    *,
    sample: Any | None = None,
    exemplar: Any | None = None,
    sample_target_value: Any | None = None,
    exemplar_target_value: Any | None = None,
    feature_matrix: Sequence[Sequence[float]] | None = None,
    target_values: Sequence[float] | None = None,
    target_name: str = "target",
    target_threshold: float | None = None,
    feature_names: Sequence[str] | None = None,
    save_dir: PathLike | None = None,
    show: bool = False,
) -> list[Path]:
    """Plot the priority of every mutable feature and categorical group.

    Args:
        priorities: ``{"numerical": ..., "categorical": ...}``.
        sample, exemplar: Feature vector(s) to mark; 2-D for several
            sample/exemplar pairs.
        sample_target_value, exemplar_target_value: Their model outputs.
        feature_matrix, target_values: Dataset rows and model targets, to
            show the data next to the priorities.
        target_name: Name of the target in labels.
        target_threshold: Decision threshold drawn on target axes.
        feature_names: Names of all columns.
        save_dir: Write one PNG per plot into this directory.
        show: Show the plots; otherwise close them.

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
            numerical_path(directory, "priority_function", idx, name)
            if directory is not None
            else None
        )
        column = None
        if feature_matrix is not None:
            rows = np.asarray(feature_matrix)
            if rows.ndim == 2 and idx < rows.shape[1]:
                column = rows[:, idx]
        plot_numerical_priority(
            idx,
            constraint,
            sample=sample,
            exemplar=exemplar,
            sample_target_value=sample_target_value,
            exemplar_target_value=exemplar_target_value,
            feature_values=column,
            target_values=target_values,
            target_name=target_name,
            target_threshold=target_threshold,
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
            group_path(directory, "priority_categorical", group_indices)
            if directory is not None
            else None
        )
        plot_categorical_priority(
            group_indices,
            mapping,
            sample=sample,
            exemplar=exemplar,
            feature_matrix=feature_matrix,
            target_values=target_values,
            target_name=target_name,
            feature_names=feature_names,
            save_path=path,
            show=show,
        )
        if path is not None:
            written.append(path)

    return written
