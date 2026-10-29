"""Search state shared by the MINLP stages."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleState:
    """Points and per-pass results of the search.

    During a search ``sample`` is the current iterate; it is restored to the
    original sample afterwards.
    """

    # original sample that you want to modify to achieve target
    sample: list = None
    # target exemplar from dataset with prediction closest to target
    target_exemplar: list = None
    # Shapley values dictionary calculated between sample and target_exemplar
    shapley_values: dict[Any, float] = None
    # This dictionary excludes categories that do not change the prediction
    # (shapley value is 0) it is done to limit number of combinations during search
    limited_priorities: dict[str, Any] = None
    # All combinations of categorical features to consider during search
    all_combinations: dict[Any, list] = None


@dataclass
class PrioritiesState:
    """Read-only views of a priorities dictionary (recomputed on every access)."""

    priorities: dict[str, Any] = field(default_factory=dict)

    @property
    def numerical_priorities(self) -> dict[str, Any]:
        return self.priorities.get("numerical", {})

    @property
    def categorical_priorities(self) -> dict[str, Any]:
        return self.priorities.get("categorical", {})

    @property
    def bounds(self) -> dict[str, Any]:
        bounds = {}
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
    def non_actionable_indices(self) -> list:
        non_actionable = []

        # Categorical groups: if only one category has a numeric (non-None) weight
        # and all other category combinations are None, mark all indices in that group
        # as non-actionable (append the individual feature indices).
        for group, mapping in self.categorical_priorities.items():
            if isinstance(mapping, dict):
                non_none_count = sum(1 for v in mapping.values() if v is not None)
                if non_none_count == 1:
                    non_actionable.append(group)

        # Numerical features: function=None marks non-actionable
        # Non-actionable features have min == max (fixed at sample value)
        for idx, val in self.numerical_priorities.items():
            if isinstance(val, dict):
                if val.get("function") is None:
                    non_actionable.append(idx)
            elif val is None:
                # Old format fallback
                non_actionable.append(idx)

        return non_actionable
