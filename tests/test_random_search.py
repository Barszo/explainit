import numpy as np
import pytest
from conftest import linear_model, mixed_model

from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.linear import basic_linear

SAMPLE = np.array([0.2, 0.4])
TARGET = 0.9


@pytest.fixture
def explainer(numerical_priorities):
    return RandomSearchExplainer(linear_model, numerical_priorities, SAMPLE, TARGET)


def _run(explainer, **kwargs):
    params = {
        "expected_counterfactuals": 3,
        "max_iterations": 5000,
        "epsilon": 0.05,
        "random_seed": 0,
    }
    params.update(kwargs)
    return explainer.generate_random_samples(**params)


class TestFilterPriorities:
    def test_actionable_features_are_kept(self, explainer, numerical_priorities):
        assert explainer.filtered_priorities["numerical"] == numerical_priorities["numerical"]

    def test_non_actionable_features_are_fixed(self):
        priorities = {
            "numerical": {
                0: None,
                1: {"min": 0.7, "max": 0.7, "function": None},
                2: {"function": None},
            },
            "categorical": {},
        }
        explainer = RandomSearchExplainer(linear_model, priorities, [0.1, 0.2, 0.3], TARGET)
        assert explainer.filtered_priorities["numerical"] == {0: 0.1, 1: 0.7, 2: 0.3}

    def test_zero_weight_categories_are_removed(self):
        priorities = {
            "numerical": {},
            "categorical": {(0, 1): {(1, 0): 1.0, (0, 1): 0}, (2, 3): {(1, 0): 0, (0, 1): 0}},
        }
        explainer = RandomSearchExplainer(linear_model, priorities, [1, 0, 1, 0], TARGET)
        assert explainer.filtered_priorities["categorical"] == {(0, 1): {(1, 0): 1.0}}


class TestPreferenceScore:
    def test_score_is_sum_of_feature_priorities(self, explainer):
        x = [0.9, 0.8]
        assert explainer.calculate_preference_score(x) == pytest.approx(2.0)
        assert explainer.calculate_preference_score([0.0, 0.0]) == pytest.approx(0.0)

    def test_categorical_weights_contribute_to_score(self, mixed_priorities):
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities, [0, 0, 0, 1], 1.0)
        assert explainer.calculate_preference_score([0.8, 0.7, 1, 0]) == pytest.approx(3.0)
        assert explainer.calculate_preference_score([0.8, 0.7, 0, 1]) == pytest.approx(2.5)
        assert explainer.calculate_preference_score([0.8, 0.7, 1, 1]) == pytest.approx(2.0)

    def test_breakdown_matches_score(self, mixed_priorities):
        mixed_priorities["numerical"][4] = {"function": None}
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities, [0, 0, 0, 1, 5], 1.0)
        x = [0.5, 0.4, 0, 1, 5]
        breakdown = explainer.get_preference_breakdown(x)
        assert breakdown["overall"] == pytest.approx(explainer.calculate_preference_score(x))
        assert breakdown["numerical"][0]["weight"] == pytest.approx(0.5)
        assert breakdown["numerical"][4] == {"value": 5, "weight": None, "actionable": False}
        assert breakdown["categorical"][(2, 3)] == {
            "combination": (0, 1),
            "weight": 0.5,
            "actionable": True,
        }


class TestGenerateRandomSamples:
    def test_counterfactuals_reach_target(self, explainer):
        cfs, preds, scores, iterations = _run(explainer)
        assert len(cfs) == len(preds) == len(scores) == len(iterations) == 3
        np.testing.assert_allclose(linear_model(np.asarray(cfs)), preds)
        assert np.all(np.abs(np.asarray(preds) - TARGET) <= 0.05)

    def test_counterfactuals_respect_bounds(self, explainer):
        cfs, *_ = _run(explainer, use_monte_carlo=False)
        assert np.all((np.asarray(cfs) >= 0.0) & (np.asarray(cfs) <= 1.0))

    def test_seed_makes_search_reproducible(self, explainer):
        first = _run(explainer, random_seed=123)
        second = _run(explainer, random_seed=123)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[3] == second[3]

    def test_unreachable_target_returns_empty_results(self, explainer):
        explainer.target = 5.0
        assert _run(explainer, max_iterations=50) == ([], [], [], [])

    def test_non_actionable_feature_is_never_changed(self):
        priorities = {
            "numerical": {
                0: {
                    "min": 0.0,
                    "max": 1.0,
                    "function": lambda x: basic_linear(x, 0.0, 1.0),
                },
                1: {"function": None},
            },
            "categorical": {},
        }
        explainer = RandomSearchExplainer(linear_model, priorities, SAMPLE, 0.6)
        cfs, *_ = _run(explainer)
        assert len(cfs) == 3
        assert np.all(np.asarray(cfs)[:, 1] == SAMPLE[1])

    @pytest.mark.parametrize("use_monte_carlo", [True, False])
    def test_forbidden_categories_are_never_sampled(self, mixed_priorities, use_monte_carlo):
        mixed_priorities["categorical"][(2, 3)][(0, 1)] = 0
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities, [0.1, 0.1, 0, 1], 1.0)
        cfs, *_ = _run(explainer, epsilon=0.1, use_monte_carlo=use_monte_carlo)
        assert len(cfs) == 3
        assert all(tuple(cf[2:]) == (1, 0) for cf in cfs)

    def test_all_categories_forbidden_raises(self, mixed_priorities):
        mixed_priorities["categorical"][(2, 3)] = {(1, 0): None, (0, 1): None}
        explainer = RandomSearchExplainer(mixed_model, mixed_priorities, [0.1, 0.1, 0, 1], 1.0)
        with pytest.raises(ValueError, match="forbidden"):
            _run(explainer)


class TestGenerateForBinary:
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
            self.classifier, priorities, np.array(sample), target_class
        )
        cfs, preds, scores, _ = explainer.generate_for_binary(
            expected_counterfactuals=3,
            max_iterations=2000,
            target_class=target_class,
            random_seed=0,
            n_candidates_per_cf=5,
        )
        assert len(cfs) == len(scores) == 3
        preds = np.asarray(preds)
        assert np.all(preds > 0.5) if target_class == 1 else np.all(preds < 0.5)
