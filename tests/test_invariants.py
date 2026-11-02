"""Rules every counterfactual must follow, checked on generated problems.

Hypothesis builds many random problems (numerical ranges, priority shapes,
immutable features, one-hot categorical groups with forbidden options, and a
reachable target) and checks that every answer the explainers return respects
the problem definition.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings, strategies as st

from explainit import (
    MINLPSearchExplainer,
    MINLPSettings,
    RandomSearchExplainer,
    basic_linear,
    exponential,
)

EPSILON = 0.02
TOL = 1e-9


@dataclass
class Problem:
    priorities: dict
    sample: np.ndarray
    target: float
    weights: np.ndarray
    category_effects: np.ndarray
    category_columns: tuple
    fixed: dict
    dataset: np.ndarray

    def model(self, X):
        X = np.asarray(X, dtype=float)
        n_num = len(self.weights)
        out = X[:, :n_num] @ self.weights + 0.2 * np.sin(3.0 * X[:, 0])
        if self.category_columns:
            out = out + X[:, list(self.category_columns)] @ self.category_effects
        return out

    def score(self, x):
        total = 0.0
        for idx, cfg in self.priorities["numerical"].items():
            if cfg["function"] is not None:
                total += float(cfg["function"](x[idx]))
        for group, mapping in self.priorities["categorical"].items():
            total += float(mapping[tuple(float(x[i]) for i in group)])
        return total

    def check(self, cf):
        """Assert that ``cf`` respects bounds, fixed features and categories."""
        cf = np.asarray(cf, dtype=float)
        assert cf.shape == self.sample.shape
        for idx, cfg in self.priorities["numerical"].items():
            assert cfg["min"] - TOL <= cf[idx] <= cfg["max"] + TOL, (idx, cf[idx], cfg)
        for idx, value in self.fixed.items():
            assert cf[idx] == pytest.approx(value), f"fixed feature {idx} changed"
        for group, mapping in self.priorities["categorical"].items():
            combo = tuple(float(cf[i]) for i in group)
            assert mapping.get(combo) is not None, f"forbidden or unknown category {combo}"


def _ramp(kind, low, high, increasing):
    if kind == "linear":
        return lambda x: basic_linear(x, low, high, increasing=increasing)
    return lambda x: exponential(x, low, high, increasing=increasing)


@st.composite
def problems(draw, allow_fixed):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    n_num = draw(st.integers(2, 4))
    n_cat = draw(st.sampled_from([0, 2, 3]))
    n_fixed = draw(st.integers(0, 1)) if allow_fixed else 0
    n_features = n_num + n_cat

    sample = np.zeros(n_features)
    sample[:n_num] = rng.uniform(0.0, 1.0, n_num)
    numerical, fixed, best_point = {}, {}, np.zeros(n_features)
    for i in range(n_num):
        if i >= n_num - n_fixed:
            fixed[i] = sample[i]
            numerical[i] = {"min": sample[i], "max": sample[i], "function": None}
            best_point[i] = sample[i]
            continue
        low = float(rng.uniform(0.0, 0.4))
        high = float(rng.uniform(low + 0.3, 1.0))
        increasing = bool(rng.random() < 0.5)
        kind = draw(st.sampled_from(["linear", "exponential"]))
        numerical[i] = {"min": low, "max": high, "function": _ramp(kind, low, high, increasing)}
        # A point with positive priority, used to make the target reachable.
        best_point[i] = rng.uniform(low + 0.1 * (high - low), high - 0.1 * (high - low))

    categorical, columns, effects = {}, (), np.zeros(0)
    if n_cat:
        columns = tuple(range(n_num, n_features))
        options = [tuple(float(v) for v in row) for row in np.eye(n_cat)]
        weights = {combo: float(rng.uniform(0.1, 1.0)) for combo in options}
        forbidden = options[int(rng.integers(n_cat))] if n_cat == 3 else None
        if forbidden is not None:
            weights[forbidden] = None
        categorical[columns] = weights
        effects = rng.uniform(-0.5, 0.5, n_cat)
        allowed = [combo for combo, w in weights.items() if w is not None]
        sample[n_num:] = options[int(rng.integers(n_cat))]
        best_point[n_num:] = allowed[int(rng.integers(len(allowed)))]

    problem = Problem(
        priorities={"numerical": numerical, "categorical": categorical},
        sample=sample,
        target=0.0,
        weights=rng.uniform(-1.0, 1.0, n_num),
        category_effects=effects,
        category_columns=columns,
        fixed=fixed,
        dataset=np.zeros((0, n_features)),
    )
    problem.target = float(problem.model([best_point])[0])

    rows = np.tile(best_point, (150, 1))
    for i, cfg in numerical.items():
        if cfg["function"] is not None:
            rows[:, i] = rng.uniform(cfg["min"], cfg["max"], 150)
    if n_cat:
        allowed = [c for c, w in categorical[columns].items() if w is not None]
        rows[:, n_num:] = np.array(allowed)[rng.integers(len(allowed), size=150)]
    problem.dataset = rows
    return problem


SETTINGS = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@SETTINGS
@given(problems(allow_fixed=True), st.booleans())
def test_random_search_answers_follow_the_rules(problem, use_monte_carlo):
    explainer = RandomSearchExplainer(
        problem.model, problem.priorities, max_iterations=3000, use_monte_carlo=use_monte_carlo
    )
    result = explainer.explain(
        problem.sample, target=problem.target, tolerance=EPSILON, n_counterfactuals=2, random_seed=0
    )
    event(f"counterfactuals found: {len(result)}/2")
    for cf, prediction, score in zip(
        result.counterfactuals, result.predictions, result.scores, strict=True
    ):
        problem.check(cf)
        assert prediction == pytest.approx(problem.model([cf])[0])
        assert abs(prediction - problem.target) <= EPSILON
        assert score == pytest.approx(problem.score(cf))


@SETTINGS
@given(problems(allow_fixed=False))
def test_minlp_answers_follow_the_rules(problem):
    explainer = MINLPSearchExplainer(
        problem.model, problem.priorities, problem.dataset, settings=MINLPSettings(max_iterations=5)
    )
    result = explainer.explain(
        problem.sample, target=problem.target, tolerance=EPSILON, random_seed=0
    )
    event(f"anchor: {result.diagnostics['exemplar_source']}")

    cf = result.counterfactuals[0]
    problem.check(cf)
    distance = abs(problem.model([cf])[0] - problem.target)
    assert bool(result.valid[0]) == (distance <= EPSILON + TOL)
    assert result.valid[0], "the target is reachable by construction"
    assert result.predictions[0] == pytest.approx(problem.model([cf])[0])
    assert result.scores[0] == pytest.approx(problem.score(cf))


@settings(deadline=None)
@given(problems(allow_fixed=True), st.integers(0, 2**31 - 1))
def test_random_search_is_deterministic_for_a_seed(problem, seed):
    def run():
        explainer = RandomSearchExplainer(problem.model, problem.priorities, max_iterations=500)
        return explainer.explain(
            problem.sample, target=problem.target, tolerance=EPSILON, n_counterfactuals=2,
            random_seed=seed,
        )  # fmt: skip

    first, second = run(), run()
    np.testing.assert_array_equal(first.counterfactuals, second.counterfactuals)
    np.testing.assert_array_equal(first.scores, second.scores)
