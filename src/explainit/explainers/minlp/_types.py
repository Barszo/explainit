"""Types shared by the MINLP stage modules."""

from collections.abc import Callable, Mapping
from typing import Any

from explainit.typing import FloatArray

# A full feature vector: the search keeps iterates as arrays or lists.
type Vector = FloatArray | list[float]
# One allowed range of a numerical feature.
type Interval = tuple[float, float]
# ``(min, max)`` of a numerical column; ``None`` means unbounded.
type Bound = tuple[float | None, float | None]
type Bounds = Mapping[int, Bound]
# One numerical priority; Stage 0 adds ``allowed_intervals`` and derived bounds.
type FeatureConfig = dict[str, Any]
type NumericalPriorities = Mapping[int, Any]
# Columns of a one-hot group, and one combination of their values.
type Group = tuple[int, ...]
type Combination = tuple[float, ...]
type CategoricalPriorities = Mapping[Group, Mapping[Combination, float | None]]
# Values of the categorical columns in one combination: ``{column: value}``.
type Assignment = dict[int, float]
# ``{"numerical": {column: phi}, "categorical": {group: phi}}``.
type ShapleyValues = dict[str, dict[Any, float]]
# A full feature vector -> a number (surrogate prediction or priority benefit).
type VectorFunction = Callable[[Vector], float]
