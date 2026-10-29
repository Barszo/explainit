"""Tests against answers that are known exactly.

These check that the algorithms are *right*, not only that they run: the best
achievable counterfactual of a linear problem comes from a linear program, and
Shapley values of simple models have closed-form values.
"""

import random

import numpy as np
import pytest
from problems import CountingModel, random_linear_problem

from explainit import MINLPSearchExplainer, RandomSearchExplainer, basic_linear, exponential
from explainit.explainers.minlp.shapley import shapley_values

SEEDS = range(8)


@pytest.mark.parametrize("seed", SEEDS)
def test_minlp_is_close_to_the_exact_optimum(seed):
    problem = random_linear_problem(seed)
    explainer = MINLPSearchExplainer(
        problem.model, problem.priorities, problem.dataset(seed=100 + seed)
    )
    result = explainer.explain(
        problem.sample, target=problem.target, tolerance=problem.epsilon, random_seed=0
    )
    cf = result.counterfactuals[0]

    assert abs(problem.model([cf])[0] - problem.target) <= problem.epsilon + 1e-9
    score, optimum = problem.score(cf), problem.optimum()
    assert score <= optimum + 1e-6, "no valid counterfactual can beat the optimum"
    assert optimum - score <= 0.05
    assert result.scores[0] == pytest.approx(score)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_search_is_close_to_the_exact_optimum(seed):
    problem = random_linear_problem(seed)
    explainer = RandomSearchExplainer(
        problem.model,
        problem.priorities,
        max_iterations=20000,
        candidates_per_counterfactual=200,
    )
    result = explainer.explain(
        problem.sample,
        target=problem.target,
        tolerance=problem.epsilon,
        n_counterfactuals=1,
        random_seed=0,
    )
    cfs, predictions, scores = result.counterfactuals, result.predictions, result.scores

    assert len(cfs) == 1
    assert abs(predictions[0] - problem.target) <= problem.epsilon
    assert scores[0] == pytest.approx(problem.score(cfs[0]))
    optimum = problem.optimum()
    assert scores[0] <= optimum + 1e-6
    assert optimum - scores[0] <= 0.2


def test_minlp_moves_a_feature_the_model_ignores_to_its_best_value():
    def model(X):
        X = np.asarray(X, dtype=float)
        return X[:, 0] + X[:, 1]

    priorities = {
        "numerical": {
            0: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.0, 1.0)},
            1: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.0, 1.0)},
            # The model ignores feature 2, so its best value costs nothing.
            2: {"min": 0.0, "max": 1.0, "function": lambda x: exponential(x, 0.0, 1.0)},
        },
        "categorical": {},
    }
    dataset = np.random.default_rng(0).uniform(0.0, 1.0, size=(100, 3))
    explainer = MINLPSearchExplainer(model, priorities, dataset)

    cf = explainer.explain(np.array([0.1, 0.1, 0.2]), target=1.0, random_seed=0).counterfactuals[0]

    assert cf[2] == pytest.approx(1.0, abs=0.01)


class TestShapleyValues:
    """``shapley_values`` attributes ``f(anchor) - f(reference)`` to the features."""

    @staticmethod
    def shapley(model, x, r, use_approximation=False, num_samples=200):
        phi, _ = shapley_values(
            model,
            np.asarray(r, dtype=float),
            np.asarray(x, dtype=float),
            numerical_indices=range(len(x)),
            categorical_groups=[],
            use_approximation=use_approximation,
            num_samples=num_samples,
            py_rng=random.Random(0),
        )
        return phi

    def test_additive_model_gives_each_feature_its_own_change(self):
        def model(X):
            X = np.asarray(X, dtype=float)
            return np.sin(X[:, 0]) + X[:, 1] ** 2 - 3.0 * X[:, 2]

        x, r = np.array([1.0, 2.0, 0.5]), np.array([0.0, -1.0, 2.0])
        phi = self.shapley(model, x, r)

        np.testing.assert_allclose(phi, [np.sin(1.0), 4.0 - 1.0, -3.0 * (0.5 - 2.0)])

    def test_interaction_is_split_by_the_shapley_formula(self):
        def model(X):
            X = np.asarray(X, dtype=float)
            return X[:, 0] * X[:, 1]

        x, r = np.array([2.0, 3.0]), np.array([1.0, 1.0])
        phi = self.shapley(model, x, r)

        expected_0 = 0.5 * ((x[0] * r[1] - r[0] * r[1]) + (x[0] * x[1] - r[0] * x[1]))
        np.testing.assert_allclose(phi, [expected_0, model([x])[0] - model([r])[0] - expected_0])

    def test_values_add_up_to_the_change_in_prediction(self):
        def model(X):
            X = np.asarray(X, dtype=float)
            return np.tanh(X[:, 0] * X[:, 1]) + X[:, 2] * X[:, 3] - X[:, 0] ** 2

        rng = np.random.default_rng(0)
        x, r = rng.normal(size=4), rng.normal(size=4)
        phi = self.shapley(model, x, r)

        assert np.sum(phi) == pytest.approx(model([x])[0] - model([r])[0])

    def test_monte_carlo_approximation_is_close_to_the_exact_value(self):
        def model(X):
            X = np.asarray(X, dtype=float)
            return X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3]

        x, r = np.array([1.0, 2.0, 1.5, -1.0]), np.zeros(4)
        exact = self.shapley(model, x, r)
        approx = self.shapley(model, x, r, use_approximation=True, num_samples=2000)

        np.testing.assert_allclose(approx, exact, atol=0.1)

    @pytest.mark.parametrize("n_features", [3, 4, 5])
    def test_exact_values_model_cost(self, n_features):
        model = CountingModel(lambda X: X.sum(axis=1))
        self.shapley(model, np.ones(n_features), np.zeros(n_features))
        # Two batched calls per feature, each over all 2**(n-1) coalitions.
        # Only 2**n coalitions are distinct, so this can be reduced n-fold.
        assert model.calls == 2 * n_features
        assert model.rows == n_features * 2**n_features
