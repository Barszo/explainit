"""Smoke tests that also run against an installed wheel outside the checkout."""

import importlib
from importlib.metadata import version

import numpy as np
import pytest

import explainit
from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.nonlinear import exponential


@pytest.mark.parametrize(
    "name",
    [
        "explainit.explainers.minlp_search",
        "explainit.explainers.random_search",
        "explainit.priorities.linear",
        "explainit.priorities.nonlinear",
        "explainit.utils.plot_styles",
        "explainit.utils.priority_plots",
    ],
)
def test_public_modules_are_installed(name):
    importlib.import_module(name)


def test_version_matches_distribution():
    assert explainit.__version__ == version("explainit")


def test_readme_random_search_example():
    def model_pred(X):
        X = np.asarray(X)
        return X[:, 0] * 0.8 + X[:, 1] * 0.2

    sample = np.array([0.2, 0.4])
    target = 0.9
    priorities = {
        "numerical": {
            0: {
                "min": 0.0,
                "max": 1.0,
                "function": lambda x: exponential(x, 0.4, 0.9, increasing=True),
            },
            1: {
                "min": 0.0,
                "max": 1.0,
                "function": lambda x: exponential(x, 0.2, 0.8, increasing=True),
            },
        },
        "categorical": {},
    }
    explainer = RandomSearchExplainer(model_pred, priorities, sample, target)
    counterfactuals, predictions, scores, iterations = explainer.generate_random_samples(
        expected_counterfactuals=3,
        max_iterations=5000,
        epsilon=0.05,
        random_seed=42,
    )
    assert len(counterfactuals) == 3
    assert len(predictions) == 3
    assert len(scores) == 3
    assert np.all(np.abs(np.asarray(predictions) - target) <= 0.05)
    np.testing.assert_allclose(model_pred(np.asarray(counterfactuals)), predictions)
    np.testing.assert_allclose(
        scores,
        [explainer.calculate_preference_score(row) for row in counterfactuals],
    )
    assert np.all((np.asarray(scores) >= 0) & (np.asarray(scores) <= 2))
    assert len(iterations) == 3
    assert np.all((np.asarray(iterations) > 0) & (np.asarray(iterations) <= 5000))
