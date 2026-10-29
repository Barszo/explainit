"""Confirmed bugs, recorded as strict expected failures.

Each test describes the correct behaviour. It fails today; when the bug is
fixed the test passes, and the strict ``xfail`` marker turns that into a
failure so the marker must be removed. Every bug is described in
``docs/known-issues.md`` under the ID used in the test's reason.
"""

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from explainit import MINLPSearchExplainer, RandomSearchExplainer, basic_linear, plot_priorities


def _bug(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def _sum_model(X):
    return np.asarray(X, dtype=float).sum(axis=1)


def _ramp(low, high, increasing=True):
    return lambda x: basic_linear(x, low, high, increasing=increasing)


class TestMINLPSearch:
    @_bug(
        "MINLP-1: MINLP replaces the user's min/max with the positive-priority region of the dataset"
    )
    def test_stays_within_the_allowed_range(self):
        priorities = {
            "numerical": {
                0: {"min": 0.3, "max": 0.8, "function": _ramp(0.3, 0.8, increasing=False)},
                1: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0)},
            },
            "categorical": {},
        }
        dataset = np.random.default_rng(0).uniform(0, 1, (200, 2))
        explainer = MINLPSearchExplainer(_sum_model, priorities, np.array([0.6, 0.2]), 1.0, dataset)
        cf = explainer.find_counterfactuals(random_seed=0)
        assert 0.3 <= cf[0] <= 0.8

    @_bug("MINLP-2: MINLP writes derived bounds into the caller's priorities dictionary")
    def test_does_not_modify_the_callers_priorities(self):
        priorities = {
            "numerical": {
                0: {"min": 0.0, "max": 1.0, "function": _ramp(0.2, 0.8)},
                1: {"min": 0.0, "max": 1.0, "function": _ramp(0.1, 0.7)},
            },
            "categorical": {},
        }
        before = copy.deepcopy(priorities)
        dataset = np.random.default_rng(0).uniform(0, 1, (100, 2))
        MINLPSearchExplainer(
            _sum_model, priorities, np.array([0.1, 0.2]), 1.0, dataset
        ).find_counterfactuals(random_seed=0)
        assert priorities == before

    @_bug("MINLP-4: find_counterfactuals_for_binary overwrites the explainer's target and epsilon")
    def test_binary_search_keeps_the_regression_settings(self, mixed_priorities, mixed_dataset):
        def classifier(X):
            X = np.asarray(X, dtype=float)
            return 1 / (1 + np.exp(-4 * (X[:, 0] + 0.5 * X[:, 1] - 0.9)))

        explainer = MINLPSearchExplainer(
            classifier, mixed_priorities, np.array([0.1, 0.2, 0.0, 1.0]), 1.0, mixed_dataset
        )
        explainer.find_counterfactuals_for_binary(target_class=1, expected_counterfactuals=1)
        assert (explainer.target, explainer.epsilon) == (1.0, 0.01)

    @_bug("MINLP-3: an anchor returned as the counterfactual keeps the anchor's immutable features")
    def test_immutable_features_keep_the_sample_value(self, mixed_priorities, mixed_dataset):
        def model(X):
            X = np.asarray(X, dtype=float)
            return X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]

        mixed_priorities["numerical"][1] = {"min": 0.2, "max": 0.2, "function": None}
        sample = np.array([0.1, 0.2, 0.0, 1.0])
        cf = MINLPSearchExplainer(
            model, mixed_priorities, sample, 0.8, mixed_dataset
        ).find_counterfactuals(random_seed=1)
        assert cf[1] == pytest.approx(0.2)


def _two_feature_priorities():
    return {
        "numerical": {
            0: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0)},
            1: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0)},
        },
        "categorical": {},
    }


class TestRandomSearch:
    @_bug("RS-1: features missing from the priorities are set to 0 instead of the sample value")
    def test_unlisted_features_keep_the_sample_value(self):
        explainer = RandomSearchExplainer(
            _sum_model, _two_feature_priorities(), np.array([0.1, 0.1, 5.0]), 6.0
        )
        cfs, *_ = explainer.generate_random_samples(1, 2000, 0.1, random_seed=0)
        assert len(cfs) == 1
        assert cfs[0][2] == 5.0

    @_bug("RS-3: a None (forbidden) categorical weight crashes the preference score")
    def test_score_of_a_forbidden_category_is_zero(self):
        priorities = {"numerical": {}, "categorical": {(0, 1): {(1, 0): 1.0, (0, 1): None}}}
        explainer = RandomSearchExplainer(_sum_model, priorities, np.array([0, 1]), 1.0)
        assert explainer.calculate_preference_score([0, 1]) == 0.0

    @_bug("RS-2: exceptions raised by the model are logged and silently skipped")
    def test_model_errors_are_raised(self):
        def broken(X):
            raise RuntimeError("model crashed")

        explainer = RandomSearchExplainer(
            broken, _two_feature_priorities(), np.array([0.1, 0.1]), 1.0
        )
        with pytest.raises(RuntimeError, match="model crashed"):
            explainer.generate_random_samples(1, 10, 0.1, random_seed=0)

    @_bug("RS-4: an invalid range (min > max) is accepted without an error")
    def test_invalid_range_is_rejected(self):
        priorities = {
            "numerical": {0: {"min": 1.0, "max": 0.0, "function": _ramp(0, 1)}},
            "categorical": {},
        }
        with pytest.raises(ValueError, match="min"):
            RandomSearchExplainer(_sum_model, priorities, np.array([0.5]), 0.5)


@_bug("PLOT-1: plotting permanently changes the global Matplotlib style")
def test_plotting_leaves_global_matplotlib_settings_unchanged():
    with plt.rc_context():
        plt.rcdefaults()
        before = dict(plt.rcParams)
        logging.disable(logging.CRITICAL)
        try:
            plot_priorities(_two_feature_priorities(), show=False)
        finally:
            logging.disable(logging.NOTSET)
        changed = [k for k in before if str(before[k]) != str(plt.rcParams[k])]
    assert changed == []
