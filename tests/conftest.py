"""Shared fixtures for the explainit test suite."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from explainit.priorities.linear import basic_linear
from explainit.priorities.nonlinear import exponential


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def linear_model(X):
    """Toy regression model: 0.8 * x0 + 0.2 * x1."""
    X = np.asarray(X, dtype=float)
    return X[:, 0] * 0.8 + X[:, 1] * 0.2


def mixed_model(X):
    """Toy regression model with two numerical and one one-hot categorical group."""
    X = np.asarray(X, dtype=float)
    return X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]


@pytest.fixture
def numerical_priorities():
    return {
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


@pytest.fixture
def mixed_priorities():
    return {
        "numerical": {
            0: {
                "min": 0.0,
                "max": 1.0,
                "function": lambda x: basic_linear(x, 0.2, 0.8, increasing=True),
            },
            1: {
                "min": 0.0,
                "max": 1.0,
                "function": lambda x: basic_linear(x, 0.1, 0.7, increasing=True),
            },
        },
        "categorical": {(2, 3): {(1, 0): 1.0, (0, 1): 0.5}},
    }


@pytest.fixture
def mixed_dataset():
    rng = np.random.default_rng(0)
    numerical = rng.uniform(0, 1, size=(40, 2))
    category = rng.integers(0, 2, size=40)
    one_hot = np.stack([category, 1 - category], axis=1)
    return np.hstack([numerical, one_hot])
