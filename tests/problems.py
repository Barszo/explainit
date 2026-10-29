"""Reusable test problems with known properties.

These helpers build small, fully specified counterfactual problems so tests can
check results against exact answers instead of only checking that code runs.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from explainit.priorities.linear import basic_linear


class CountingModel:
    """Wrap a model and count how often, and on how many rows, it is called."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self.rows = 0

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        self.calls += 1
        self.rows += X.shape[0]
        return self.fn(X)


@dataclass
class LinearProblem:
    """Linear model with linear (ramp) priorities over [0, 1].

    Both the constraint and the objective are linear, so the best achievable
    preference score is the solution of a linear program.
    """

    weights: np.ndarray
    increasing: np.ndarray
    sample: np.ndarray
    target: float
    epsilon: float = 0.01
    priorities: dict = field(init=False)

    def __post_init__(self):
        self.priorities = {
            "numerical": {
                i: {
                    "min": 0.0,
                    "max": 1.0,
                    "function": lambda x, inc=bool(inc): basic_linear(x, 0.0, 1.0, increasing=inc),
                }
                for i, inc in enumerate(self.increasing)
            },
            "categorical": {},
        }

    def model(self, X):
        return np.asarray(X, dtype=float) @ self.weights

    def score(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(np.where(self.increasing, x, 1.0 - x)))

    def optimum(self):
        """Best preference score of any valid counterfactual (exact)."""
        cost = -np.where(self.increasing, 1.0, -1.0)
        result = linprog(
            cost,
            A_ub=np.vstack([self.weights, -self.weights]),
            b_ub=[self.target + self.epsilon, -(self.target - self.epsilon)],
            bounds=[(0.0, 1.0)] * len(self.weights),
        )
        assert result.success, result.message
        return self.score(result.x)

    def dataset(self, n_rows=200, seed=0):
        return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n_rows, len(self.weights)))


def random_linear_problem(seed, n_features=3):
    """A reachable LinearProblem whose target lies well inside the reachable range."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.2, 1.0, n_features) * rng.choice([-1.0, 1.0], n_features)
    low = np.sum(np.minimum(weights, 0.0))
    high = np.sum(np.maximum(weights, 0.0))
    return LinearProblem(
        weights=weights,
        increasing=rng.random(n_features) < 0.5,
        sample=rng.uniform(0.0, 1.0, n_features),
        target=float(rng.uniform(low + 0.2 * (high - low), high - 0.2 * (high - low))),
    )
