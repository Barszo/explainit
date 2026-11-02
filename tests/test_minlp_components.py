"""Unit tests for the building blocks of the MINLP search.

Each stage is a plain function or small class, so it can be checked on its
own with tiny inputs.
"""

import random

import numpy as np
import pytest

from explainit.explainers.minlp.bounds import TrustRegion, derive_bounds_and_intervals
from explainit.explainers.minlp.combinations import limit_combinations
from explainit.explainers.minlp.incumbent import Incumbents
from explainit.explainers.minlp.scoring import (
    feature_priority,
    priority_benefit,
    priority_peaks,
    total_weight,
)
from explainit.explainers.minlp.shapley import shapley_values
from explainit.explainers.minlp.surrogate import (
    categorical_contribution,
    reachable_range,
    surrogate_prediction,
    unit_coefficients,
)
from explainit.priorities.linear import basic_linear


def _linear(x0, x1, increasing=True):
    return lambda x: basic_linear(x, x0, x1, increasing=increasing)


class TestTrustRegion:
    def test_disabled_region_returns_the_bounds(self):
        bounds = {0: (0.0, 1.0)}
        assert TrustRegion().limit(bounds, [0.5]) == bounds
        assert TrustRegion().grow() is False

    def test_region_is_a_fraction_of_the_span_around_the_center(self):
        region = TrustRegion(fraction=0.25)
        assert region.limit({0: (0.0, 2.0), 1: (None, 1.0)}, [1.0, 0.3]) == {
            0: (0.5, 1.5),
            1: (None, 1.0),
        }

    def test_center_outside_the_bounds_is_clamped(self):
        assert TrustRegion(fraction=0.1).limit({0: (0.0, 1.0)}, [5.0]) == {0: (0.9, 1.0)}

    def test_grow_and_shrink_respect_the_limits(self):
        region = TrustRegion(fraction=0.5, minimum=0.2, maximum=0.8)
        assert region.grow() is True
        assert region.fraction == 0.8
        assert region.grow() is False
        region.shrink()
        region.shrink()
        assert region.fraction == pytest.approx(0.2)


class TestDeriveBounds:
    def test_bounds_follow_the_positive_priority_region(self):
        priorities = {
            0: {"min": 0.0, "max": 1.0, "function": _linear(0.4, 0.6)},
            1: {"min": 0.0, "max": 1.0, "function": None},
        }
        dataset = np.array([[0.0, 0.0], [1.0, 1.0]])
        derive_bounds_and_intervals(priorities, dataset, [0.5, 0.7], str, grid_size=101)

        assert priorities[0]["min"] == pytest.approx(0.41)
        assert priorities[0]["max"] == pytest.approx(1.0)
        assert priorities[1]["allowed_intervals"] == [(0.0, 0.0)]

    def test_feature_without_positive_priority_is_rejected(self):
        priorities = {0: {"min": 0.0, "max": 1.0, "function": lambda x: 0.0}}
        with pytest.raises(ValueError, match="no positive-priority region"):
            derive_bounds_and_intervals(priorities, np.array([[0.0], [1.0]]), [0.5], str)


class TestScoring:
    NUMERICAL = {  # noqa: RUF012
        0: {"min": 0.0, "max": 1.0, "function": _linear(0.0, 1.0)},
        1: {"min": 0.0, "max": 1.0, "function": None},
    }
    CATEGORICAL = {(2, 3): {(1.0, 0.0): 1.0, (0.0, 1.0): 0.25}}  # noqa: RUF012

    def test_total_weight_sums_priorities_and_category_weights(self):
        assert total_weight([0.5, 9.0, 0.0, 1.0], self.NUMERICAL, self.CATEGORICAL) == 0.75

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ([1.5, 0.0, 1.0, 0.0], "outside allowed range"),
            ([0.5, 0.0, 1.0, 1.0], "Invalid categorical"),
        ],
    )
    def test_disallowed_values_have_no_benefit(self, values, message):
        with pytest.raises(ValueError, match=message):
            total_weight(values, self.NUMERICAL, self.CATEGORICAL)
        assert priority_benefit(values, self.NUMERICAL, self.CATEGORICAL) is None

    def test_feature_priority_and_peaks(self):
        assert feature_priority(self.NUMERICAL[0], 0.25) == pytest.approx(0.25)
        assert feature_priority(self.NUMERICAL[1], 0.25) == 0.0
        assert priority_peaks(self.NUMERICAL, grid_size=11) == {0: (1.0, 1.0)}


class TestSurrogate:
    def test_surrogate_is_exact_at_the_linearisation_point(self):
        sample, anchor = np.array([0.0, 1.0, 1.0, 0.0]), np.array([2.0, 1.0, 0.0, 1.0])
        coefficients, _ = unit_coefficients({0: 4.0, 1: 3.0}, sample, anchor, [0, 1], [])
        assert coefficients == {0: 2.0, 1: 0.0}  # feature 1 does not move

        shapley = {"numerical": {0: 4.0, 1: 3.0}, "categorical": {(2, 3): 0.5}}
        priorities = {"numerical": {0: {}, 1: {}}, "categorical": {(2, 3): {}}}
        h = lambda x: surrogate_prediction(  # noqa: E731
            x, coefficients, shapley, sample, anchor, priorities, basic_prediction=1.0
        )
        assert h(sample) == 1.0
        assert h(anchor) == pytest.approx(1.0 + 4.0 + 0.5)
        with pytest.raises(ValueError, match="Invalid categorical combination"):
            h(np.array([0.0, 1.0, 1.0, 1.0]))

    def test_categorical_contribution_of_an_assignment(self):
        sample, anchor = [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        groups, phi = [(2, 3)], {(2, 3): 0.7}
        assert categorical_contribution({2: 0.0, 3: 1.0}, groups, phi, sample, anchor) == 0.7
        assert categorical_contribution({2: 1.0, 3: 0.0}, groups, phi, sample, anchor) == 0.0

    def test_reachable_range_uses_interval_arithmetic(self):
        low, high = reachable_range(
            1.0, {0: 2.0, 1: -1.0, 2: 5.0}, {0: (0.0, 1.0), 1: (0.0, 2.0), 2: (0.0, 1.0)},
            [0.5, 1.0, 0.0], immutable={2},
        )  # fmt: skip
        assert (low, high) == (pytest.approx(-1.0), pytest.approx(3.0))


class TestShapley:
    def test_categorical_group_is_one_unit(self):
        def model(X):
            X = np.asarray(X, dtype=float)
            return 2.0 * X[:, 0] + 3.0 * X[:, 1] - X[:, 2]

        phi, values = shapley_values(
            model, [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0], [(1, 2)], py_rng=random.Random(0)
        )
        np.testing.assert_allclose(phi, [2.0, -4.0])
        assert values == {"numerical": {0: 2.0}, "categorical": {(1, 2): -4.0}}


class TestCombinations:
    def test_groups_without_influence_keep_the_sample_category(self):
        priorities = {
            "numerical": {},
            "categorical": {
                (0, 1): {(1.0, 0.0): 1.0, (0.0, 1.0): 0.5},
                (2, 3): {(1.0, 0.0): 1.0, (0.0, 1.0): 0.5},
            },
        }
        sample, anchor = [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]
        limited, combinations = limit_combinations(
            priorities, {(0, 1): 0.3, (2, 3): 0.0}, sample, anchor
        )

        assert limited["categorical"][(2, 3)] == {(1.0, 0.0): 1.0}
        assert list(combinations.values()) == [
            {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
            {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0},
        ]
        assert priorities["categorical"][(2, 3)] == {(1.0, 0.0): 1.0, (0.0, 1.0): 0.5}


class TestIncumbents:
    @staticmethod
    def evaluation(feasible, distance, priority):
        return {"feasible": feasible, "distance": distance, "priority": priority}

    def test_feasible_candidates_are_ranked_by_priority(self):
        incumbents = Incumbents()
        assert incumbents.offer([1], self.evaluation(False, 0.5, 2.0))
        assert incumbents.offer([2], self.evaluation(False, 0.2, 0.0))
        assert not incumbents.offer([3], self.evaluation(False, 0.3, 9.0))
        assert incumbents.infeasible == [2]

        assert incumbents.offer([4], self.evaluation(True, 0.01, 1.0))
        assert not incumbents.offer([5], self.evaluation(True, 0.0, 0.5))
        assert incumbents.offer([6], self.evaluation(True, 0.02, 1.5))
        assert not incumbents.offer([7], self.evaluation(False, 0.0, 9.0))
        assert (incumbents.feasible, incumbents.feasible_source) == ([6], "optimiser")
        assert incumbents.describe() == ("1.5000", "0.0200")
