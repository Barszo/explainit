"""Stage 4.1: prune categorical combinations with Shapley values.

A categorical group whose Shapley value is zero cannot help reach the target,
so it keeps the sample's category. Every other group may take either the
sample's or the anchor's category. The search then runs once per combination.
"""

import copy
import itertools


def limit_combinations(priorities, shapley_categorical, sample, anchor):
    """Restrict each categorical group to at most the sample's and the anchor's category.

    Args:
        priorities: Full priorities ``{"numerical": ..., "categorical": ...}``.
        shapley_categorical: ``{group: phi}`` Shapley values of the groups.
        sample: Current point.
        anchor: Anchor exemplar.

    Returns:
        tuple: ``(limited_priorities, combinations)``. ``limited_priorities``
        is a copy of ``priorities`` with only the kept categories;
        ``combinations`` maps an id to a ``{column: value}`` assignment, one
        per element of the product of the kept categories.
    """
    limited = copy.deepcopy(priorities)
    allowed_by_group = []

    for group, weights in priorities.get("categorical", {}).items():
        sample_values = tuple(float(sample[idx]) for idx in group)
        anchor_values = tuple(float(anchor[idx]) for idx in group)
        shap_val = shapley_categorical.get(group, 0.0)

        kept_weights = {}
        kept = []
        if shap_val == 0:
            if sample_values in weights:
                kept_weights[sample_values] = weights[sample_values]
                kept.append(sample_values)
        else:
            if sample_values in weights:
                kept_weights[sample_values] = weights[sample_values]
                kept.append(sample_values)
            if anchor_values in weights and anchor_values != sample_values:
                kept_weights[anchor_values] = weights[anchor_values]
                kept.append(anchor_values)

        limited["categorical"][group] = kept_weights
        allowed_by_group.append(kept)

    combinations = {}
    groups = list(priorities.get("categorical", {}))
    for combination_id, values_per_group in enumerate(itertools.product(*allowed_by_group)):
        assignment = {}
        for group_index, group in enumerate(groups):
            for i, idx in enumerate(group):
                assignment[idx] = values_per_group[group_index][i]
        combinations[combination_id] = assignment

    return limited, combinations
