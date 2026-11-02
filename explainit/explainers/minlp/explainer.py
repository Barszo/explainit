"""``MINLPSearchExplainer``: Shapley-guided counterfactual search."""

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from explainit.explainers.minlp.search import MINLPSearch
from explainit.explanation import Explanation
from explainit.targets import resolve_target
from explainit.utils.priority_plots import plot_priorities as _plot_priorities


@dataclass(frozen=True)
class MINLPSettings:
    """Tuning options of the MINLP search. The defaults suit most problems.

    Attributes:
        max_iterations: Maximum refinement passes (regression targets).
        patience: Stop after this many passes without improvement.
        approximate_shapley: Estimate Shapley values from random coalitions
            instead of enumerating all of them (faster for many features).
        shapley_samples: Coalitions per feature for the estimate.
        anchor_tolerance: How close a dataset row's prediction must be to the
            target to serve as the anchor.
        anchor_sampling_iterations: Random samples tried when no dataset row
            can serve as the anchor.
        trust_region: Limit how far one pass may move each feature.
        trust_region_init: Initial step limit, as a fraction of each feature's
            allowed span.
        trust_region_min: Smallest step limit.
        trust_region_max: Largest step limit.
        residual_correction: Correct the surrogate's target by the error
            measured at a rejected step.
        restoration: Pull candidates that just miss the target back into it.
        peak_lock_in: After the search, move features to their most preferred
            values where other features can compensate.
        peak_lock_in_max_sweeps: Maximum peak lock-in sweeps.
        return_when_fails: Return the closest candidate when the target is not
            reached (marked as not valid); otherwise return none.
        binary_max_iterations: Optimiser iterations per categorical
            combination for ``target_class`` searches.
    """

    max_iterations: int = 10
    patience: int = 5
    approximate_shapley: bool = False
    shapley_samples: int = 200
    anchor_tolerance: float = 0.01
    anchor_sampling_iterations: int = 10000
    trust_region: bool = True
    trust_region_init: float = 0.25
    trust_region_min: float = 0.02
    trust_region_max: float = 1.0
    residual_correction: bool = True
    restoration: bool = True
    peak_lock_in: bool = True
    peak_lock_in_max_sweeps: int = 3
    return_when_fails: bool = True
    binary_max_iterations: int = 100


class MINLPSearchExplainer:
    """Preference-aware counterfactual search guided by Shapley values.

    Linearises the model around the current point with Shapley values towards
    an anchor near the target, and optimises the priorities under that linear
    model, checking every candidate with the real model. See
    ``docs/how-it-works.md``.

    Args:
        model: Callable mapping a 2-D array (or a list of rows) to one
            prediction per row.
        priorities: ``{"numerical": ..., "categorical": ...}``; see the README.
        dataset: Reference rows, e.g. the training data, used to find anchors.
        settings: Tuning options; see :class:`MINLPSettings`.
        feature_names: Names used in log messages.
        workflow_logger: Optional logger that receives a short summary of
            each search.
    """

    def __init__(
        self,
        model,
        priorities,
        dataset,
        *,
        settings: MINLPSettings | None = None,
        feature_names: Sequence[str] | None = None,
        workflow_logger=None,
    ):
        self.model = model
        self.priorities = priorities
        self.dataset = np.asarray(dataset)
        self.settings = settings if settings is not None else MINLPSettings()
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.workflow_logger = workflow_logger

    def explain(
        self,
        x,
        *,
        target=None,
        tolerance=0.01,
        target_class=None,
        threshold=0.5,
        random_seed=None,
    ) -> Explanation:
        """Find a counterfactual for the sample ``x``.

        Give either ``target`` (a prediction value, reached within
        ``tolerance``) or ``target_class`` (1: probability above ``threshold``;
        0: below it).

        Returns:
            Explanation: For a ``target``, one counterfactual (or none); its
            ``valid`` flag says whether it reaches the target, and
            ``diagnostics`` describes the search (stop reason, history, anchor,
            ...). For a ``target_class``, one counterfactual per categorical
            combination that crossed the threshold.
        """
        goal = resolve_target(target, tolerance, target_class, threshold)
        sample = np.array(x, dtype=float)
        search = self._search(sample, target=goal.value, tolerance=goal.tolerance)
        if goal.is_class:
            return self._explain_class(search, sample, goal)

        s = self.settings
        cf = search.find_counterfactuals(
            shap_approx=s.approximate_shapley,
            num_samples=s.shapley_samples,
            max_iterations=s.max_iterations,
            patience=s.patience,
            return_when_fails=s.return_when_fails,
            fallback_random_max_iterations=s.anchor_sampling_iterations,
            peak_lock_in=s.peak_lock_in,
            peak_lock_in_max_sweeps=s.peak_lock_in_max_sweeps,
            random_seed=random_seed,
            trust_region=s.trust_region,
            trust_region_init=s.trust_region_init,
            trust_region_min=s.trust_region_min,
            trust_region_max=s.trust_region_max,
            residual_correction=s.residual_correction,
            restoration=s.restoration,
        )
        diagnostics = {**search.last_search_result, "target": goal, "settings": asdict(s)}
        found = [] if cf is None else [cf]
        return Explanation.from_lists(
            sample,
            found,
            [diagnostics["prediction"]] * len(found),
            [diagnostics["priority_score"]] * len(found),
            [diagnostics["reached_target"]] * len(found),
            method="minlp_search",
            diagnostics=diagnostics,
        )

    def probe_feasibility(
        self, x, *, target, tolerance=0.01, max_iterations=5000, random_seed=None
    ) -> dict:
        """Check whether random points of the allowed region can reach ``target``.

        Separates *feasibility* failures (the allowed region cannot reach the
        target) from *convergence* failures (it can, but the search did not).

        Returns:
            dict: ``feasible_within_epsilon``, ``min_pred_distance`` and
            ``iterations_run``.
        """
        search = self._search(np.array(x, dtype=float), target=target, tolerance=tolerance)
        return search.probe_allowed_region_feasibility(
            max_iterations=max_iterations, random_seed=random_seed
        )

    def plot_priorities(self, x=None, *, feature_names=None, save_dir=None, show=True):
        """Plot the priorities, marking ``x`` if given.

        Pass ``save_dir`` to also write PNG files, and ``show=False`` to skip
        opening the plots.
        """
        return _plot_priorities(
            self.priorities,
            sample=x,
            feature_names=feature_names if feature_names is not None else self.feature_names,
            save_dir=save_dir,
            show=show,
        )

    def _search(self, sample, *, target, tolerance) -> MINLPSearch:
        """A fresh search with its own copy of the priorities."""
        return MINLPSearch(
            self.model,
            copy.deepcopy(self.priorities),
            sample,
            target,
            self.dataset,
            target_exemplar_epsilon=self.settings.anchor_tolerance,
            epsilon=tolerance,
            workflow_logger=self.workflow_logger,
            feature_names=self.feature_names,
        )

    def _explain_class(self, search, sample, goal) -> Explanation:
        s = self.settings
        cfs, predictions, scores, combinations = search.find_counterfactuals_for_binary(
            target_class=goal.target_class,
            threshold=goal.threshold,
            max_iterations=s.binary_max_iterations,
            shap_approx=s.approximate_shapley,
            num_samples=s.shapley_samples,
        )
        return Explanation.from_lists(
            sample,
            cfs,
            predictions,
            scores,
            [True] * len(cfs),
            method="minlp_search",
            diagnostics={
                "combination_numbers": combinations,
                "target": goal,
                "settings": asdict(s),
            },
        )
