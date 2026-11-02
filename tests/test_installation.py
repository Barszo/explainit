"""Smoke tests that also run against an installed wheel outside the checkout."""

import importlib
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pytest

import explainit
from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.nonlinear import exponential


@pytest.mark.parametrize(
    "name",
    [
        "explainit.explainers.minlp",
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
    explainer = RandomSearchExplainer(model_pred, priorities, max_iterations=5000)
    result = explainer.explain(
        sample, target=target, tolerance=0.05, n_counterfactuals=3, random_seed=42
    )
    assert len(result) == 3
    assert np.all(np.abs(result.predictions - target) <= 0.05)
    np.testing.assert_allclose(model_pred(result.counterfactuals), result.predictions)
    np.testing.assert_allclose(
        result.scores, [explainer.preference_score(row) for row in result.counterfactuals]
    )
    assert np.all((result.scores >= 0) & (result.scores <= 2))


def test_import_does_not_configure_logging():
    code = (
        "import logging, explainit; "
        "root = logging.getLogger(); "
        "assert not root.handlers, root.handlers; "
        "assert root.level == logging.WARNING, root.level"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_exports():
    from explainit.explainers.minlp import MINLPSearchExplainer
    from explainit.priorities.linear import basic_linear

    assert explainit.MINLPSearchExplainer is MINLPSearchExplainer
    assert explainit.RandomSearchExplainer is RandomSearchExplainer
    assert explainit.basic_linear is basic_linear
    assert explainit.exponential is exponential
    for name in explainit.__all__:
        assert hasattr(explainit, name)
