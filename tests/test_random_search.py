import random

import numpy as np
import pytest
from conftest import linear_model, mixed_model

from explainit import Explanation, RandomSearchExplainer
from explainit.explainers.random_search import filter_priorities
from explainit.priorities.linear import basic_linear

SAMPLE = np.array([0.2, 0.4])
TARGET = 0.9


@pytest.fixture
def explainer(numerical_priorities):
    return RandomSearchExplainer(linear_model, numerical_priorities, max_iterations=5000)


def _explain(explainer, x=SAMPLE, **kwargs):
    params = {"target": TARGET, "tolerance": 0.05, "n_counterfactuals": 3, "random_seed": 0}
    params.update(kwargs)
    return explainer.explain(x, **params)


class TestFilterPriorities:
    def test_actionable_features_are_kept(self, numerical_priorities):
        filtered = filter_priorities(numerical_priorities, SAMPLE)
        assert filtered["numerical"] == numerical_priorities["numerical"]

    def test_non_actionable_features_are_fixed(self):
        priorities = {
            "numerical": {
                0: None,
                1: {"min": 0.7, "max": 0.7, "function": None},
                2: {"function": None},
            },
            "categorical": {},
        }
        filtered = filter_priorities(priorities, [0.1, 0.2, 0.3])
        assert filtered["numerical"] == {0: 0.1, 1: 0.7, 2: 0.3}

    def test_zero_weight_categories_are_removed(self):
        priorities = {
            "numerical": {},
            "categorical": {(0, 1): {(1, 0): 1.0, (0, 1): 0}, (2, 3): {(1, 0): 0, (0, 1): 0}},
        }
        assert filter_priorities(priorities, [1, 0, 1, 0])["categorical"] == {(0, 1): {(1, 0): 1.0}}


class TestPreferenceScore:
    def test_score_is_sum_of_feature_priorities(self, explainer):
        assert explainer.preference_score([0.9, 0.8]) == pytest.approx(2.0)
        assert explainer.preference_score([0.0, 0.0]) == pytest.approx(0.0)

    def test_categorical_weights_contribute_to_score(self, mixed_priorities):
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities)
        assert explainer.preference_score([0.8, 0.7, 1, 0]) == pytest.approx(3.0)
        assert explainer.preference_score([0.8, 0.7, 0, 1]) == pytest.approx(2.5)
        assert explainer.preference_score([0.8, 0.7, 1, 1]) == pytest.approx(2.0)

    def test_breakdown_matches_score(self, mixed_priorities):
        mixed_priorities["numerical"][4] = {"function": None}
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities)
        x = [0.5, 0.4, 0, 1, 5]
        breakdown = explainer.preference_breakdown(x)
        assert breakdown["overall"] == pytest.approx(explainer.preference_score(x))
        assert breakdown["numerical"][0]["weight"] == pytest.approx(0.5)
        assert breakdown["numerical"][4] == {"value": 5, "weight": None, "actionable": False}
        assert breakdown["categorical"][(2, 3)] == {
            "combination": (0, 1),
            "weight": 0.5,
            "actionable": True,
        }


class TestExplain:
    def test_result_is_an_explanation_that_reaches_the_target(self, explainer):
        result = _explain(explainer)
        assert isinstance(result, Explanation)
        assert len(result) == 3
        assert result.counterfactuals.shape == (3, 2)
        assert result.method == "random_search"
        assert result.valid.all()
        np.testing.assert_allclose(linear_model(result.counterfactuals), result.predictions)
        assert np.all(np.abs(result.predictions - TARGET) <= 0.05)
        np.testing.assert_allclose(
            result.scores, [explainer.preference_score(cf) for cf in result.counterfactuals]
        )
        np.testing.assert_allclose(result.changes, result.counterfactuals - SAMPLE)
        assert len(result.diagnostics["iterations"]) == 3

    def test_counterfactuals_respect_bounds(self, numerical_priorities):
        explainer = RandomSearchExplainer(linear_model, numerical_priorities, use_monte_carlo=False)
        result = _explain(explainer)
        assert np.all((result.counterfactuals >= 0.0) & (result.counterfactuals <= 1.0))

    def test_seed_makes_search_reproducible(self, explainer):
        first = _explain(explainer, random_seed=123)
        second = _explain(explainer, random_seed=123)
        np.testing.assert_array_equal(first.counterfactuals, second.counterfactuals)
        assert first.diagnostics["iterations"] == second.diagnostics["iterations"]

    def test_explainer_can_be_reused_for_other_samples(self, explainer):
        other = np.array([0.6, 0.1])
        first = _explain(explainer, x=other, random_seed=1)
        _explain(explainer, x=SAMPLE, random_seed=7)
        again = _explain(explainer, x=other, random_seed=1)
        np.testing.assert_array_equal(first.counterfactuals, again.counterfactuals)
        np.testing.assert_array_equal(first.sample, other)

    def test_seeded_search_leaves_global_random_state_untouched(self, explainer):
        np_state = np.random.get_state()[1].copy()
        py_state = random.getstate()
        _explain(explainer, random_seed=123)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state)
        assert random.getstate() == py_state

    def test_top_n_keeps_most_preferable(self, explainer):
        all_found = _explain(explainer, n_counterfactuals=5)
        top = _explain(explainer, n_counterfactuals=5, top_n=2)
        assert top.scores.tolist() == sorted(all_found.scores, reverse=True)[:2]
        assert len(top) == 2

    def test_more_candidates_per_counterfactual_improve_scores(self, numerical_priorities):
        def mean_score(candidates):
            explainer = RandomSearchExplainer(
                linear_model, numerical_priorities, candidates_per_counterfactual=candidates
            )
            return _explain(explainer).scores.mean()

        assert mean_score(10) > mean_score(1)

    def test_unreachable_target_returns_an_empty_explanation(self, numerical_priorities):
        explainer = RandomSearchExplainer(linear_model, numerical_priorities, max_iterations=50)
        result = _explain(explainer, target=5.0)
        assert len(result) == 0
        assert result.counterfactuals.shape == (0, 2)

    def test_non_actionable_feature_is_never_changed(self):
        priorities = {
            "numerical": {
                0: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.0, 1.0)},
                1: {"function": None},
            },
            "categorical": {},
        }
        result = _explain(RandomSearchExplainer(linear_model, priorities), target=0.6)
        assert len(result) == 3
        assert np.all(result.counterfactuals[:, 1] == SAMPLE[1])

    @pytest.mark.parametrize("use_monte_carlo", [True, False])
    def test_forbidden_categories_are_never_sampled(self, mixed_priorities, use_monte_carlo):
        mixed_priorities["categorical"][(2, 3)][(0, 1)] = 0
        explainer = RandomSearchExplainer(
            mixed_model, mixed_priorities, use_monte_carlo=use_monte_carlo
        )
        result = _explain(explainer, x=[0.1, 0.1, 0, 1], target=1.0, tolerance=0.1)
        assert len(result) == 3
        assert all(tuple(cf[2:]) == (1, 0) for cf in result.counterfactuals)

    def test_all_categories_forbidden_raises(self, mixed_priorities):
        mixed_priorities["categorical"][(2, 3)] = {(1, 0): None, (0, 1): None}
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities)
        with pytest.raises(ValueError, match="forbidden"):
            _explain(explainer, x=[0.1, 0.1, 0, 1], target=1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"target": 1.0, "target_class": 1},
            {"target_class": 2},
            {"target": 1.0, "tolerance": -1},
        ],
    )
    def test_invalid_target_arguments_are_rejected(self, explainer, kwargs):
        with pytest.raises(ValueError, match=r"target|tolerance"):
            explainer.explain(SAMPLE, **kwargs)


class TestExplainClass:
    @staticmethod
    def classifier(X):
        return 1.0 / (1.0 + np.exp(-10.0 * (linear_model(X) - 0.5)))

    @pytest.mark.parametrize(("target_class", "sample"), [(1, [0.1, 0.1]), (0, [0.9, 0.9])])
    def test_counterfactuals_cross_threshold(self, target_class, sample):
        increasing = target_class == 1
        priorities = {
            "numerical": {
                i: {
                    "min": 0.0,
                    "max": 1.0,
                    "function": lambda x, inc=increasing: basic_linear(x, 0.0, 1.0, inc),
                }
                for i in range(2)
            },
            "categorical": {},
        }
        explainer = RandomSearchExplainer(
            self.classifier, priorities, max_iterations=2000, candidates_per_counterfactual=5
        )
        result = explainer.explain(
            np.array(sample), target_class=target_class, n_counterfactuals=3, random_seed=0
        )
        assert len(result) == 3
        if target_class == 1:
            assert np.all(result.predictions > 0.5)
        else:
            assert np.all(result.predictions < 0.5)


def test_to_frame(explainer):
    pytest.importorskip("pandas")
    frame = _explain(explainer).to_frame(["income", "savings"])
    assert list(frame.columns) == ["income", "savings", "prediction", "score", "valid"]
    assert len(frame) == 3
    with pytest.raises(ValueError, match="feature names"):
        _explain(explainer).to_frame(["only_one"])
