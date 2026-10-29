import random

import numpy as np
import pytest

from explainit.priorities.linear import basic_linear
from explainit.sampling import sample_category, sample_numeric_value

INCREASING = {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.0, 1.0)}


class TestSampleNumericValue:
    def test_density_follows_the_priority(self):
        rng = np.random.RandomState(0)
        values = np.array([sample_numeric_value(INCREASING, rng=rng) for _ in range(4000)])
        assert np.all((values >= 0.0) & (values <= 1.0))
        # Density 2x on [0, 1]: mean 2/3, a quarter of the mass below 0.5.
        assert values.mean() == pytest.approx(2 / 3, abs=0.02)
        assert np.mean(values < 0.5) == pytest.approx(0.25, abs=0.03)

    def test_grid_fallback_follows_the_priority(self):
        rng = np.random.RandomState(0)
        values = np.array(
            [sample_numeric_value(INCREASING, max_tries=0, rng=rng) for _ in range(4000)]
        )
        assert values.mean() == pytest.approx(2 / 3, abs=0.02)

    def test_zero_priority_falls_back_to_uniform(self):
        constraint = {"min": 2.0, "max": 4.0, "function": lambda x: np.zeros_like(x)}
        rng = np.random.RandomState(0)
        values = np.array(
            [sample_numeric_value(constraint, max_tries=5, rng=rng) for _ in range(2000)]
        )
        assert values.mean() == pytest.approx(3.0, abs=0.05)

    def test_same_seed_same_values(self):
        def draw():
            rng = np.random.RandomState(7)
            return [sample_numeric_value(INCREASING, rng=rng) for _ in range(20)]

        assert draw() == draw()


class TestSampleCategory:
    WEIGHTS = {(1, 0, 0): 3.0, (0, 1, 0): 1.0, (0, 0, 1): 0}  # noqa: RUF012

    def test_weighted_picks_are_proportional_to_the_weights(self):
        np_rng, py_rng = np.random.RandomState(0), random.Random(0)
        picks = [
            sample_category((0, 1, 2), self.WEIGHTS, True, np_rng, py_rng) for _ in range(4000)
        ]
        assert picks.count((1, 0, 0)) / len(picks) == pytest.approx(0.75, abs=0.03)
        assert (0, 0, 1) not in picks

    def test_uniform_picks_ignore_the_weights_but_not_forbidden_options(self):
        np_rng, py_rng = np.random.RandomState(0), random.Random(0)
        picks = [
            sample_category((0, 1, 2), self.WEIGHTS, False, np_rng, py_rng) for _ in range(4000)
        ]
        assert picks.count((1, 0, 0)) / len(picks) == pytest.approx(0.5, abs=0.03)
        assert (0, 0, 1) not in picks

    @pytest.mark.parametrize("weights", [{(1, 0): 0, (0, 1): 0}, {(1, 0): None, (0, 1): -1.0}])
    def test_all_forbidden_raises(self, weights):
        with pytest.raises(ValueError, match="forbidden"):
            sample_category((0, 1), weights, True, np.random.RandomState(0), random.Random(0))
