"""Search state shared by the MINLP stages."""

from dataclasses import dataclass, field
from typing import Any

from explainit.explainers.minlp._types import (
    Assignment,
    Bound,
    ShapleyValues,
    Vector,
)


class _Unset(Exception):
    """A search value was read before the stage that computes it ran."""


@dataclass
class SampleState:
    """Points and per-pass results of the search.

    During a search ``sample`` is the current iterate; it is restored to the
    original sample afterwards. The other values are filled in by the stages
    (anchor in Stage 1, Shapley values in Stage 3, combinations and
    coefficients in Stage 4); reading one earlier is an error.
    """

    sample: Vector
    _target_exemplar: Vector | None = field(default=None, repr=False)
    _shapley_values: ShapleyValues | None = field(default=None, repr=False)
    _limited_priorities: dict[str, Any] | None = field(default=None, repr=False)
    _all_combinations: dict[int, Assignment] | None = field(default=None, repr=False)
    # Per-unit surrogate coefficients; ``None`` until Stage 4 (callers check).
    shap_coeffs: dict[int, float] | None = None

    @property
    def target_exemplar(self) -> Vector:
        """The anchor the surrogate is linearised towards (Stage 1)."""
        value = self._target_exemplar
        if value is None:
            raise _Unset("target_exemplar is not computed yet")
        return value

    @target_exemplar.setter
    def target_exemplar(self, value: Vector) -> None:
        self._target_exemplar = value

    @property
    def shapley_values(self) -> ShapleyValues:
        """Shapley values between the current point and the anchor (Stage 3)."""
        value = self._shapley_values
        if value is None:
            raise _Unset("shapley_values is not computed yet")
        return value

    @shapley_values.setter
    def shapley_values(self, value: ShapleyValues) -> None:
        self._shapley_values = value

    def has_shapley_values(self) -> bool:
        return self._shapley_values is not None

    @property
    def limited_priorities(self) -> dict[str, Any]:
        """Priorities with the categorical options pruned in Stage 4.1."""
        value = self._limited_priorities
        if value is None:
            raise _Unset("limited_priorities is not computed yet")
        return value

    @limited_priorities.setter
    def limited_priorities(self, value: dict[str, Any]) -> None:
        self._limited_priorities = value

    @property
    def all_combinations(self) -> dict[int, Assignment]:
        """Categorical combinations searched in this pass (Stage 4.1)."""
        value = self._all_combinations
        if value is None:
            raise _Unset("all_combinations is not computed yet")
        return value

    @all_combinations.setter
    def all_combinations(self, value: dict[int, Assignment]) -> None:
        self._all_combinations = value


@dataclass
class PrioritiesState:
    """Read-only views of a priorities dictionary (recomputed on every access)."""

    priorities: dict[str, Any] = field(default_factory=dict)

    @property
    def numerical_priorities(self) -> dict[int, Any]:
        """``{column: FeatureConfig}``; legacy formats may hold ``None`` for immutable features."""
        numerical: dict[int, Any] = self.priorities.get("numerical", {})
        return numerical

    @property
    def categorical_priorities(self) -> dict[Any, Any]:
        categorical: dict[Any, Any] = self.priorities.get("categorical", {})
        return categorical

    @property
    def bounds(self) -> dict[int, Bound]:
        bounds: dict[int, Bound] = {}
        for idx, val in self.numerical_priorities.items():
            if isinstance(val, dict):
                min_val = val.get("min", None)
                max_val = val.get("max", None)
                bounds[idx] = (min_val, max_val)
            else:
                # This shouldn't happen with new format, but keep as fallback
                bounds[idx] = (None, None)
        return bounds

    @property
    def non_actionable_indices(self) -> list[Any]:
        """Immutable numerical columns, and categorical groups with a single allowed option."""
        non_actionable: list[Any] = []

        for group, mapping in self.categorical_priorities.items():
            if isinstance(mapping, dict):
                non_none_count = sum(1 for v in mapping.values() if v is not None)
                if non_none_count == 1:
                    non_actionable.append(group)

        for idx, val in self.numerical_priorities.items():
            if isinstance(val, dict):
                if val.get("function") is None:
                    non_actionable.append(idx)
            elif val is None:
                # Old format fallback
                non_actionable.append(idx)

        return non_actionable
