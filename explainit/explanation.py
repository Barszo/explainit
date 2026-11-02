"""The result of explaining one sample."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from explainit.typing import FloatArray

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class Explanation:
    """Counterfactuals found for one sample.

    Attributes:
        sample: The explained sample.
        counterfactuals: One counterfactual per row, shape ``(n, n_features)``.
        predictions: Model prediction for each counterfactual.
        scores: Preference score of each counterfactual (higher is better;
            NaN when it cannot be computed).
        valid: Whether each counterfactual reaches the target.
        method: Name of the search method.
        diagnostics: Method-specific details about the search.
    """

    sample: FloatArray
    counterfactuals: FloatArray
    predictions: FloatArray
    scores: FloatArray
    valid: npt.NDArray[np.bool_]
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_lists(
        cls,
        sample: npt.ArrayLike,
        counterfactuals: Sequence[npt.ArrayLike],
        predictions: Sequence[float],
        scores: Sequence[float | None],
        valid: Sequence[bool],
        method: str,
        diagnostics: dict[str, Any],
    ) -> "Explanation":
        """Build an explanation from per-counterfactual lists."""
        sample = np.asarray(sample, dtype=float)
        rows = np.asarray(counterfactuals, dtype=float).reshape(-1, sample.shape[-1])
        return cls(
            sample=sample,
            counterfactuals=rows,
            predictions=np.asarray(predictions, dtype=float).reshape(-1),
            scores=np.array([np.nan if s is None else s for s in scores], dtype=float),
            valid=np.asarray(valid, dtype=bool).reshape(-1),
            method=method,
            diagnostics=diagnostics,
        )

    def __len__(self) -> int:
        return len(self.counterfactuals)

    @property
    def changes(self) -> FloatArray:
        """``counterfactuals - sample``: how each feature has to change."""
        return self.counterfactuals - self.sample

    def to_frame(self, feature_names: Sequence[str] | None = None) -> "pd.DataFrame":
        """The counterfactuals as a :class:`pandas.DataFrame`.

        One row per counterfactual with a column per feature, then
        ``prediction``, ``score`` and ``valid``. Requires pandas.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("Explanation.to_frame() needs pandas: pip install pandas") from exc
        n_features = self.counterfactuals.shape[1]
        columns = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(n_features)]
        )
        if len(columns) != n_features:
            raise ValueError(f"expected {n_features} feature names, got {len(columns)}")
        frame = pd.DataFrame(self.counterfactuals, columns=columns)
        frame["prediction"] = self.predictions
        frame["score"] = self.scores
        frame["valid"] = self.valid
        return frame
