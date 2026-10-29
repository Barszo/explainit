import logging
import random

import numpy as np
import pytest
from conftest import mixed_model

from explainit.explainers.minlp import MINLPSearchExplainer
from explainit.explainers.minlp.intervals import (
    extract_positive_intervals,
    in_allowed_intervals,
    project_to_allowed_intervals,
)
from explainit.explainers.minlp.linear_program import solve_linear_constraint
from explainit.explainers.minlp.state import PrioritiesState

SAMPLE = np.array([0.1, 0.2, 0.0, 1.0])
TARGET = 1.0
EPSILON = 0.01

# Attributes read by downstream experiment code (explainit-experiment-2);
# removing or renaming them is a breaking change.
LAST_SEARCH_RESULT_KEYS = {
    "reached_target",
    "distance",
    "iterations_run",
    "stop_reason",
    "best_cf",
    "history",
    "cf_source",
    "priority_score",
    "anchor_priority_score",
    "priority_gain_vs_anchor",
    "exemplar_source",
    "exemplar_pred_distance",
}
EXEMPLAR_SOURCES = {"dataset_priority_filtered", "dataset_actionable", "random_in_range"}


@pytest.fixture
def explainer(mixed_priorities, mixed_dataset):
    return MINLPSearchExplainer(mixed_model, mixed_priorities, SAMPLE, TARGET, mixed_dataset)


class TestPrioritiesState:
    def test_bounds_and_non_actionable_indices(self):
        state = PrioritiesState(
            priorities={
                "numerical": {
                    0: {"min": 0.0, "max": 1.0, "function": lambda x: x},
                    1: {"min": 0.5, "max": 0.5, "function": None},
                },
                "categorical": {
                    (2, 3): {(1, 0): 1.0, (0, 1): None},
                    (4, 5): {(1, 0): 1.0, (0, 1): 0.3},
                },
            }
        )
        assert state.bounds == {0: (0.0, 1.0), 1: (0.5, 0.5)}
        assert state.non_actionable_indices == [(2, 3), 1]


class TestIntervalHelpers:
    INTERVALS = ((0.0, 0.2), (0.5, 0.8))

    @pytest.mark.parametrize(
        ("value", "expected"), [(0.1, True), (0.5, True), (0.3, False), (0.9, False)]
    )
    def test_in_allowed_intervals(self, value, expected):
        assert in_allowed_intervals(value, self.INTERVALS) is expected

    def test_empty_intervals_allow_nothing(self):
        assert in_allowed_intervals(0.1, []) is False

    @pytest.mark.parametrize(
        ("value", "expected"), [(0.1, 0.1), (0.3, 0.2), (0.4, 0.5), (1.0, 0.8)]
    )
    def test_project_to_nearest_allowed_value(self, value, expected):
        projected = project_to_allowed_intervals(value, self.INTERVALS)
        assert projected == pytest.approx(expected)

    def test_extract_positive_intervals(self):
        x = np.linspace(0.0, 1.0, 11)
        y = np.where((x < 0.25) | ((x > 0.55) & (x < 0.85)), 1.0, 0.0)
        intervals = extract_positive_intervals(x, y)
        np.testing.assert_allclose(intervals, [(0.0, 0.2), (0.6, 0.8)])


class TestLinearProgram:
    def test_finds_point_on_hyperplane_within_bounds(self):
        result = solve_linear_constraint([1.0, 2.0], 1.5, [(0.0, 1.0), (0.0, 1.0)])
        assert result["success"]
        x = result["solution"]
        assert np.dot([1.0, 2.0], x) == pytest.approx(1.5)
        assert np.all((x >= 0.0) & (x <= 1.0))

    def test_infeasible_target_fails(self):
        result = solve_linear_constraint([1.0, 1.0], 5.0, [(0.0, 1.0), (0.0, 1.0)])
        assert not result["success"]
        assert result["solution"] is None

    def test_mismatched_bounds_are_rejected(self):
        result = solve_linear_constraint([1.0, 1.0], 1.0, [(0.0, 1.0)])
        assert not result["success"]

    def test_all_zero_coefficients(self):
        bounds = [(0.0, 1.0), (2.0, 4.0)]
        ok = solve_linear_constraint([0.0, 0.0], 0.0, bounds)
        assert ok["success"]
        np.testing.assert_allclose(ok["solution"], [0.5, 3.0])
        assert not solve_linear_constraint([0.0, 0.0], 1.0, bounds)["success"]


class TestFindCounterfactuals:
    def test_reaches_target_within_bounds(self, explainer):
        cf = np.asarray(explainer.find_counterfactuals(random_seed=1), dtype=float)
        assert abs(mixed_model([cf])[0] - TARGET) <= EPSILON + 1e-9
        assert np.all((cf[:2] >= 0.0) & (cf[:2] <= 1.0))
        assert tuple(cf[2:]) in {(1.0, 0.0), (0.0, 1.0)}

    def test_last_search_result_contract(self, explainer):
        explainer.find_counterfactuals(random_seed=1)
        result = explainer.last_search_result
        assert set(result) >= LAST_SEARCH_RESULT_KEYS
        assert result["reached_target"] is True
        assert result["cf_source"] in {"anchor", "optimiser"}
        assert explainer.exemplar_source in EXEMPLAR_SOURCES
        assert explainer.exemplar_pred_distance == pytest.approx(result["exemplar_pred_distance"])

    def test_seed_makes_search_reproducible(self, mixed_priorities, mixed_dataset):
        def run():
            explainer = MINLPSearchExplainer(
                mixed_model, mixed_priorities, SAMPLE, TARGET, mixed_dataset
            )
            return explainer.find_counterfactuals(random_seed=7)

        np.testing.assert_allclose(run(), run())

    def test_seeded_search_leaves_global_random_state_untouched(self, explainer):
        np_state = np.random.get_state()[1].copy()
        py_state = random.getstate()
        explainer.find_counterfactuals(shap_approx=True, num_samples=20, random_seed=3)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state)
        assert random.getstate() == py_state

    def test_approximate_shapley_values(self, explainer):
        cf = explainer.find_counterfactuals(shap_approx=True, num_samples=50, random_seed=2)
        assert abs(mixed_model([cf])[0] - TARGET) <= EPSILON + 1e-9

    @pytest.mark.parametrize(
        "options",
        [
            {"trust_region": False},
            {"residual_correction": False, "restoration": False},
            {"peak_lock_in": False},
        ],
    )
    def test_optional_stages_can_be_disabled(self, explainer, options):
        cf = explainer.find_counterfactuals(random_seed=1, **options)
        assert abs(mixed_model([cf])[0] - TARGET) <= EPSILON + 1e-9

    def test_workflow_logger_receives_progress(self, mixed_priorities, mixed_dataset, caplog):
        workflow_logger = logging.getLogger("tests.workflow")
        explainer = MINLPSearchExplainer(
            mixed_model,
            mixed_priorities,
            SAMPLE,
            TARGET,
            mixed_dataset,
            workflow_logger=workflow_logger,
            feature_names=["income", "savings", "owner", "renter"],
        )
        with caplog.at_level(logging.INFO, logger="tests.workflow"):
            explainer.find_counterfactuals(random_seed=1)
        assert any(r.name == "tests.workflow" for r in caplog.records)

    def test_unreachable_target_raises(self, mixed_priorities, mixed_dataset):
        explainer = MINLPSearchExplainer(mixed_model, mixed_priorities, SAMPLE, 5.0, mixed_dataset)
        explainer._fallback_random_max_iterations = 200
        with pytest.raises(Exception, match="Could not randomly generate"):
            explainer.find_counterfactuals(random_seed=1, return_when_fails=False)


class TestFindCounterfactualsForBinary:
    @staticmethod
    def classifier(X):
        return 1.0 / (1.0 + np.exp(-4.0 * (mixed_model(X) - 0.9)))

    def test_counterfactuals_cross_threshold(self, mixed_priorities, mixed_dataset):
        explainer = MINLPSearchExplainer(
            self.classifier, mixed_priorities, SAMPLE, 1, mixed_dataset
        )
        cfs, preds, scores, counts = explainer.find_counterfactuals_for_binary(
            target_class=1, threshold=0.5
        )
        assert len(cfs) == len(preds) == len(scores) == len(counts) > 0
        assert np.all(np.asarray(preds) > 0.5)
        np.testing.assert_allclose(self.classifier(np.asarray(cfs, dtype=float)), preds)
