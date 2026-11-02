"""Types used in explainit's public API.

Use them to annotate your own code, e.g.::

    from explainit.typing import Model, Priorities
"""

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
"""A NumPy array of floats."""

type Model = Callable[[Any], Any]
"""A model: takes a 2-D array (one row per sample) and returns one prediction per row.

explainit also passes a list of rows, so wrap models that need an array, e.g.
``lambda X: clf.predict_proba(np.asarray(X))[:, 1]``.
"""

type PriorityFunction = Callable[[Any], Any]
"""Desirability of a numerical value, in ``[0, 1]``; must accept floats and arrays."""

type NumericalPriority = Mapping[str, Any]
"""``{"min": float, "max": float, "function": PriorityFunction | None}``.

``function=None`` marks an immutable feature, fixed at ``min``.
"""

type CategoricalWeights = Mapping[tuple[float, ...], float | None]
"""Weight of each combination of a one-hot group; ``0`` or ``None`` forbids it."""

type Priorities = Mapping[str, Mapping[Any, Any]]
"""``{"numerical": {column: NumericalPriority}, "categorical": {columns: CategoricalWeights}}``."""

type PathLike = str | os.PathLike[str]
"""A file or directory path."""

__all__ = [
    "CategoricalWeights",
    "FloatArray",
    "Model",
    "NumericalPriority",
    "PathLike",
    "Priorities",
    "PriorityFunction",
]
