"""Stage 3: Shapley values between the current point and the anchor.

Shapley values attribute the change in prediction ``f(anchor) - f(reference)``
to the features. Numerical features are one unit each; every one-hot
categorical group is a single unit, so its columns always switch together.
"""

import itertools
from collections.abc import Iterable
from math import factorial

import numpy as np

from explainit._random import PythonRandom
from explainit.explainers.minlp._types import Group, ShapleyValues, Vector
from explainit.logging_config import logger
from explainit.typing import FloatArray, Model

# A Shapley unit's value: a number (numerical feature) or a tuple (categorical group).
type Unit = float | tuple[float, ...]


def shapley_values(
    model: Model,
    reference: Vector,
    anchor: Vector,
    numerical_indices: Iterable[int],
    categorical_groups: Iterable[Group],
    use_approximation: bool = False,
    num_samples: int = 200,
    py_rng: PythonRandom = None,
) -> tuple[FloatArray, ShapleyValues]:
    """Shapley values of moving from ``reference`` towards ``anchor``.

    Args:
        model: Callable mapping a 2-D array to one prediction per row.
        reference: Feature vector the coalitions start from (the current point).
        anchor: Feature vector whose values a coalition takes over.
        numerical_indices: Column indices of the numerical units, in order.
        categorical_groups: Column-index tuples of the categorical units.
        use_approximation: Average ``num_samples`` random coalitions per unit
            instead of enumerating all ``2**(n-1)`` of them.
        num_samples: Coalitions per unit for the approximation.
        py_rng: ``random.Random``-like generator used by the approximation.

    Returns:
        tuple: ``(phi, values)`` where ``phi`` is an array with one value per
        unit (numerical units first) and ``values`` is
        ``{"numerical": {index: phi}, "categorical": {group: phi}}``.
    """
    numerical_indices = list(numerical_indices)
    cat_groups = [list(group) for group in categorical_groups]

    def create_feature_mapping() -> tuple[dict[int, int], int]:
        """Map every original column to its Shapley unit."""
        feature_map: dict[int, int] = {}
        shapley_idx = 0

        for num_idx in numerical_indices:
            feature_map[num_idx] = shapley_idx
            shapley_idx += 1

        for cat_group in cat_groups:
            for cat_idx in cat_group:
                feature_map[cat_idx] = shapley_idx
            shapley_idx += 1

        return feature_map, shapley_idx

    def create_consolidated_vectors(
        original_r: Vector, original_x: Vector, feature_map: dict[int, int], total_units: int
    ) -> tuple[list[Unit], list[Unit]]:
        """One entry per unit; categorical units hold a tuple of column values."""
        consolidated_r: list[Unit] = [0] * total_units
        consolidated_x: list[Unit] = [0] * total_units

        for num_idx in numerical_indices:
            shapley_idx = feature_map[num_idx]
            consolidated_r[shapley_idx] = original_r[num_idx]
            consolidated_x[shapley_idx] = original_x[num_idx]

        for cat_group in cat_groups:
            shapley_idx = feature_map[cat_group[0]]
            consolidated_r[shapley_idx] = tuple(original_r[i] for i in cat_group)
            consolidated_x[shapley_idx] = tuple(original_x[i] for i in cat_group)

        return consolidated_r, consolidated_x

    def expand_to_original_format(
        consolidated_vector: list[Unit], feature_map: dict[int, int], original_length: int
    ) -> list[Unit]:
        """Inverse of :func:`create_consolidated_vectors`."""
        expanded: list[Unit] = [0] * original_length

        for num_idx in numerical_indices:
            shapley_idx = feature_map[num_idx]
            expanded[num_idx] = consolidated_vector[shapley_idx]

        for cat_group in cat_groups:
            shapley_idx = feature_map[cat_group[0]]
            cat_values = consolidated_vector[shapley_idx]
            if isinstance(cat_values, tuple):
                for i, cat_idx in enumerate(cat_group):
                    expanded[cat_idx] = cat_values[i]
            else:
                for cat_idx in cat_group:
                    expanded[cat_idx] = cat_values

        return expanded

    def coalition_batch(
        subsets: list[set[int]],
        consolidated_r: list[Unit],
        consolidated_x: list[Unit],
        feature_map: dict[int, int],
        original_length: int,
    ) -> FloatArray:
        """Rows that take the anchor's values for the units in each subset."""
        z_batch = []
        for S in subsets:
            z_consolidated = consolidated_r.copy()
            for i in S:
                z_consolidated[i] = consolidated_x[i]
            z_batch.append(expand_to_original_format(z_consolidated, feature_map, original_length))
        return np.array(z_batch)

    def marginal_contributions(
        subsets: list[set[int]],
        subsets_with_i: list[set[int]],
        consolidated_x: list[Unit],
        consolidated_r: list[Unit],
        f: Model,
        feature_map: dict[int, int],
        original_length: int,
    ) -> FloatArray:
        z_without = coalition_batch(
            subsets, consolidated_r, consolidated_x, feature_map, original_length
        )
        z_with = coalition_batch(
            subsets_with_i, consolidated_r, consolidated_x, feature_map, original_length
        )
        pred_without = f(z_without)
        pred_with = f(z_with)
        if pred_without.ndim > 1:
            pred_without = pred_without.flatten()
        if pred_with.ndim > 1:
            pred_with = pred_with.flatten()
        diff: FloatArray = pred_with - pred_without
        return diff

    def exact_value(
        i: int,
        consolidated_x: list[Unit],
        consolidated_r: list[Unit],
        f: Model,
        n: int,
        feature_map: dict[int, int],
        original_length: int,
    ) -> float:
        others = [j for j in range(n) if j != i]

        subsets: list[set[int]] = []
        subsets_with_i: list[set[int]] = []
        weights: list[float] = []

        logger.info(
            "[Stage 3] Exact Shapley for unit %d/%d: enumerating %d subsets (O(2^(n-1)) work).",
            i + 1,
            n,
            2 ** len(others),
        )

        for k in range(len(others) + 1):
            for combination in itertools.combinations(others, k):
                S = set(combination)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                subsets.append(S)
                subsets_with_i.append(S | {i})
                weights.append(weight)

        diff = marginal_contributions(
            subsets,
            subsets_with_i,
            consolidated_x,
            consolidated_r,
            f,
            feature_map,
            original_length,
        )
        return float(np.sum(np.array(weights) * diff))

    def approximate_value(
        i: int,
        consolidated_x: list[Unit],
        consolidated_r: list[Unit],
        f: Model,
        n: int,
        feature_map: dict[int, int],
        original_length: int,
        num_samples: int,
    ) -> float:
        others = [j for j in range(n) if j != i]

        logger.info(
            "[Stage 3] Approximate Shapley for unit %d/%d: averaging %d "
            "random subsets (Monte-Carlo estimator).",
            i + 1,
            n,
            num_samples,
        )

        subsets: list[set[int]] = []
        subsets_with_i: list[set[int]] = []
        for _ in range(num_samples):
            subset_size = py_rng.randint(0, len(others))
            if subset_size == 0:
                S = set()
            elif subset_size == len(others):
                S = set(others)
            else:
                S = set(py_rng.sample(others, subset_size))
            subsets.append(S)
            subsets_with_i.append(S | {i})

        diff = marginal_contributions(
            subsets,
            subsets_with_i,
            consolidated_x,
            consolidated_r,
            f,
            feature_map,
            original_length,
        )
        return float(np.mean(diff))

    original_length = len(reference)
    feature_map, total_units = create_feature_mapping()
    consolidated_r, consolidated_x = create_consolidated_vectors(
        reference, anchor, feature_map, total_units
    )

    logger.info(
        "[Stage 3] Consolidating features for Shapley: %d original cols -> "
        "%d units (numerical kept 1:1, each one-hot group merged).",
        original_length,
        total_units,
    )
    logger.info(
        "[Stage 3] Numerical features tracked: %d | Categorical groups: %d",
        len(numerical_indices),
        len(cat_groups),
    )

    if use_approximation:
        phi = [
            approximate_value(
                i,
                consolidated_x,
                consolidated_r,
                model,
                total_units,
                feature_map,
                original_length,
                num_samples,
            )
            for i in range(total_units)
        ]
    else:
        phi = [
            exact_value(
                i,
                consolidated_x,
                consolidated_r,
                model,
                total_units,
                feature_map,
                original_length,
            )
            for i in range(total_units)
        ]

    values: ShapleyValues = {"numerical": {}, "categorical": {}}
    unit = 0
    for num_idx in numerical_indices:
        values["numerical"][num_idx] = phi[unit]
        unit += 1
    for cat_group in cat_groups:
        values["categorical"][tuple(cat_group)] = phi[unit]
        unit += 1

    return np.array(phi, dtype=float), values
