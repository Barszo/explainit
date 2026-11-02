"""Confirmed bugs, recorded as strict expected failures.

Each test describes the correct behaviour and fails today with the specific
error the bug causes (``raises=``), so an unrelated error is not mistaken for
the bug. When a bug is fixed, its test passes and the strict ``xfail`` turns
that into a failure, so the marker must be removed. Every bug is described in
``docs/known-issues.md`` under the ID in the reason.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from explainit import MINLPSearchExplainer, RandomSearchExplainer, basic_linear, plot_priorities

ASSERTION_FAILS = AssertionError
DOES_NOT_RAISE = pytest.fail.Exception


def _bug(reason, raises):
    return pytest.mark.xfail(strict=True, reason=reason, raises=raises)


def _sum_model(X):
    return np.asarray(X, dtype=float).sum(axis=1)


def _ramp(low, high, increasing=True):
    return lambda x: basic_linear(x, low, high, increasing=increasing)


def _two_feature_priorities():
    return {
        "numerical": {
            0: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0)},
            1: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0)},
        },
        "categorical": {},
    }


class TestMINLPSearch:
    @_bug("MINLP-1: the user's min/max are replaced by the dataset range", ASSERTION_FAILS)
    def test_stays_within_the_allowed_range(self):
        priorities = {
            "numerical": {
                0: {"min": 0.3, "max": 0.8, "function": _ramp(0.3, 0.8, increasing=False)},
                1: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0)},
            },
            "categorical": {},
        }
        dataset = np.random.default_rng(0).uniform(0, 1, (200, 2))
        explainer = MINLPSearchExplainer(_sum_model, priorities, dataset)
        cf = explainer.explain(np.array([0.6, 0.2]), target=1.0, random_seed=0).counterfactuals[0]
        assert 0.3 <= cf[0] <= 0.8

    @_bug("MINLP-3: a returned anchor keeps its own immutable values", ASSERTION_FAILS)
    def test_immutable_features_keep_the_sample_value(self, mixed_priorities, mixed_dataset):
        def model(X):
            X = np.asarray(X, dtype=float)
            return X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]

        mixed_priorities["numerical"][1] = {"min": 0.2, "max": 0.2, "function": None}
        sample = np.array([0.1, 0.2, 0.0, 1.0])
        explainer = MINLPSearchExplainer(model, mixed_priorities, mixed_dataset)
        cf = explainer.explain(sample, target=0.8, random_seed=1).counterfactuals[0]
        assert cf[1] == pytest.approx(0.2)


class TestRandomSearch:
    @_bug("RS-1: features missing from the priorities are set to 0", ASSERTION_FAILS)
    def test_unlisted_features_keep_the_sample_value(self):
        explainer = RandomSearchExplainer(
            _sum_model, _two_feature_priorities(), max_iterations=2000
        )
        result = explainer.explain(
            np.array([0.1, 0.1, 5.0]), target=6.0, tolerance=0.1, n_counterfactuals=1, random_seed=0
        )
        assert len(result) == 1
        assert result.counterfactuals[0][2] == 5.0

    @_bug("RS-3: a None (forbidden) category weight crashes the score", TypeError)
    def test_score_of_a_forbidden_category_is_zero(self):
        priorities = {"numerical": {}, "categorical": {(0, 1): {(1, 0): 1.0, (0, 1): None}}}
        explainer = RandomSearchExplainer(_sum_model, priorities)
        assert explainer.preference_score([0, 1]) == 0.0

    @_bug("RS-2: exceptions raised by the model are silently skipped", DOES_NOT_RAISE)
    def test_model_errors_are_raised(self):
        def broken(X):
            raise RuntimeError("model crashed")

        explainer = RandomSearchExplainer(broken, _two_feature_priorities(), max_iterations=10)
        with pytest.raises(RuntimeError, match="model crashed"):
            explainer.explain(np.array([0.1, 0.1]), target=1.0, n_counterfactuals=1, random_seed=0)

    @_bug("RS-4: an invalid range (min > max) is accepted", DOES_NOT_RAISE)
    def test_invalid_range_is_rejected(self):
        priorities = {
            "numerical": {0: {"min": 1.0, "max": 0.0, "function": _ramp(0, 1)}},
            "categorical": {},
        }
        with pytest.raises(ValueError, match="min"):
            RandomSearchExplainer(_sum_model, priorities)


@_bug("PLOT-1: plotting permanently changes the global Matplotlib style", ASSERTION_FAILS)
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
