from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from explainit._random import make_rngs
from explainit.explainers.minlp.anchor import (
    closest_row,
    filter_rows,
    generate_in_range,
    probe_feasibility,
)
from explainit.explainers.minlp.bounds import TrustRegion, derive_bounds_and_intervals
from explainit.explainers.minlp.combinations import limit_combinations
from explainit.explainers.minlp.intervals import (
    allowed_interval_constraint_value,
    clip_to_allowed,
    in_allowed_intervals,
    project_to_allowed_intervals,
)
from explainit.explainers.minlp.optimizer import optimise_combination
from explainit.explainers.minlp.refinement import (
    RefinementContext,
    evaluate_candidate,
    peak_lock_in as lock_in_priority_peaks,
    restore_to_band,
    select_best_candidate,
)
from explainit.explainers.minlp.scoring import (
    feature_priority,
    priority_benefit,
    priority_peaks,
    total_weight,
)
from explainit.explainers.minlp.shapley import shapley_values
from explainit.explainers.minlp.surrogate import (
    categorical_contribution,
    reachable_range,
    surrogate_prediction,
    unit_coefficients,
)
from explainit.explainers.minlp.warm_start import solve_warm_starts, warm_start_health
from explainit.logging_config import logger
from explainit.utils.priority_plots import plot_priorities as _plot_priorities


@dataclass
class SampleState:
    # original sample that you want to modify to achieve target
    sample: list = None
    # target exemplar from dataset with prediction closest to target
    target_exemplar: list = None
    # Shapley values dictionary calculated between sample and target_exemplar
    shapley_values: dict[Any, float] = None
    # This dictionary excludes categories that do not change the prediction
    # (shapley value is 0) it is done to limit number of combinations during search
    limited_priorities: dict[str, Any] = None
    # All combinations of categorical features to consider during search
    all_combinations: dict[Any, list] = None


@dataclass
class PrioritiesState:
    # TODO: at the end, make sure that the attributes are calculated once only when invoked, and then only when priorities are changed
    priorities: dict[str, Any] = field(default_factory=dict)

    @property
    def numerical_priorities(self) -> dict[str, Any]:
        return self.priorities.get("numerical", {})

    @property
    def categorical_priorities(self) -> dict[str, Any]:
        return self.priorities.get("categorical", {})

    @property
    def bounds(self) -> dict[str, Any]:
        bounds = {}
        for idx, val in self.numerical_priorities.items():
            if isinstance(val, dict):
                min_val = val.get("min", None)
                max_val = val.get("max", None)
                bounds[idx] = (min_val, max_val)
            else:
                # This shouldn't happen with new format, but keep as fallback
                bounds[idx] = (None, None)
        return bounds

    @property
    def non_actionable_indices(self) -> list:
        non_actionable = []

        # Categorical groups: if only one category has a numeric (non-None) weight
        # and all other category combinations are None, mark all indices in that group
        # as non-actionable (append the individual feature indices).
        for group, mapping in self.categorical_priorities.items():
            if isinstance(mapping, dict):
                non_none_count = sum(1 for v in mapping.values() if v is not None)
                if non_none_count == 1:
                    non_actionable.append(group)

        # Numerical features: function=None marks non-actionable
        # Non-actionable features have min == max (fixed at sample value)
        for idx, val in self.numerical_priorities.items():
            if isinstance(val, dict):
                if val.get("function") is None:
                    non_actionable.append(idx)
            elif val is None:
                # Old format fallback
                non_actionable.append(idx)

        return non_actionable


class MINLPSearchExplainer:
    def __init__(
        self,
        model_pred,
        priorities,
        sample,
        target,
        dataset,
        target_exemplar_epsilon=0.01,
        epsilon=0.01,
        workflow_logger=None,
        feature_names: Sequence[str] | None = None,
    ):
        self.sample_state = SampleState(sample=sample)
        self.priorities_state = PrioritiesState(priorities=priorities)
        # TODO: model_pred needs to be prepared before to be used here. It should be changed so that it would be more universal
        self.model_pred = model_pred
        self.target_exemplar_epsilon = target_exemplar_epsilon  # Finds element in dataset which prediction is closest to target
        # There is a section in the code below
        self.target = target
        self.dataset = dataset
        self.epsilon = epsilon
        self.workflow_logger = workflow_logger
        self.feature_names = list(feature_names) if feature_names is not None else None
        # Cached argmax of every actionable priority function (Stage 7).
        self._priority_peaks_cache = None
        # Records which strategy produced the target exemplar for the last run
        # (see ``_stage1_find_exemplar``): one of ``dataset_priority_filtered``,
        # ``dataset_actionable``, ``random_in_range`` or ``none``.
        self.exemplar_source = None
        # |model_pred(target_exemplar) - target| for the anchor of the last run.
        self.exemplar_pred_distance = None
        # Warm-start (Stage 4 LP) health captured on the first refinement pass.
        self._warm_start_info = None
        # Exception text if a refinement pass aborted (stop_reason=search_failed).
        self._last_search_exception = None
        self.last_search_result = {}
        self._fallback_random_max_iterations = 10000
        # Seed for the only stochastic part of the search (the Stage 1
        # fallback anchor); set per run by ``find_counterfactuals``.
        self._random_seed = None
        # Random streams for the stochastic steps (see ``make_rngs``); replaced
        # per run by ``find_counterfactuals``.
        self._np_rng, self._py_rng = make_rngs(None)
        # Step limit for each optimisation pass (Stages 4-5).
        self._trust_region = TrustRegion()
        # Additive surrogate correction: the linear constraint aims at
        # ``target - _target_correction`` so the surrogate reproduces the last
        # measured model residual.
        self._target_correction = 0.0
        self._residual_correction_enabled = False
        self._pass_targets = {}
        self._restoration_enabled = False

    def _workflow_log(self, message, *args):
        if self.workflow_logger is not None:
            self.workflow_logger.info(message, *args)

    def _feature_label(self, idx: int) -> str:
        if self.feature_names is not None and 0 <= idx < len(self.feature_names):
            return str(self.feature_names[idx])
        return f"feature_{idx}"

    def _log_workflow_initial_and_bounds(self):
        initial_prediction = float(self.model_pred([self.sample_state.sample])[0])
        self._workflow_log("")
        self._workflow_log("==== MINLP WORKFLOW SUMMARY ====")
        self._workflow_log("1) Initial prediction: %.6f", initial_prediction)
        self._workflow_log("2) Target: %.6f", float(self.target))
        self._workflow_log("")
        self._workflow_log("3) Bounds per feature vs dataset min/max:")
        n_rows = int(self.dataset.shape[0])
        global_allowed_mask = np.ones(n_rows, dtype=bool)

        for idx in sorted(self.priorities_state.numerical_priorities.keys()):
            dataset_min = float(np.min(self.dataset[:, idx]))
            dataset_max = float(np.max(self.dataset[:, idx]))
            cfg = self.priorities_state.numerical_priorities.get(idx, {})
            bounds_min, bounds_max = self.priorities_state.bounds[idx]
            allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None

            allowed_min = dataset_min if bounds_min is None else float(bounds_min)
            allowed_max = dataset_max if bounds_max is None else float(bounds_max)
            dataset_span = dataset_max - dataset_min
            allowed_span = allowed_max - allowed_min
            coverage_pct = (
                100.0 if abs(dataset_span) < 1e-12 else (allowed_span / dataset_span) * 100.0
            )
            if allowed_intervals:
                feature_allowed_mask = np.array(
                    [
                        in_allowed_intervals(float(v), allowed_intervals)
                        for v in self.dataset[:, idx]
                    ],
                    dtype=bool,
                )
            else:
                feature_allowed_mask = np.logical_and(
                    self.dataset[:, idx] >= allowed_min,
                    self.dataset[:, idx] <= allowed_max,
                )
            feature_allowed_pct = (
                100.0 * float(np.sum(feature_allowed_mask)) / float(n_rows) if n_rows else 0.0
            )
            global_allowed_mask = np.logical_and(global_allowed_mask, feature_allowed_mask)

            self._workflow_log(
                "   - %s (idx=%d): bounds=[%.6f, %.6f] | dataset=[%.6f, %.6f] | allowed space=%.2f%% | allowed points=%.2f%%",
                self._feature_label(idx),
                idx,
                allowed_min,
                allowed_max,
                dataset_min,
                dataset_max,
                coverage_pct,
                feature_allowed_pct,
            )
        global_allowed_pct = (
            100.0 * float(np.sum(global_allowed_mask)) / float(n_rows) if n_rows else 0.0
        )
        self._workflow_log("")
        self._workflow_log(
            "Global allowed samples (all numerical-feature bounds simultaneously): %.2f%% (%d/%d)",
            global_allowed_pct,
            int(np.sum(global_allowed_mask)),
            n_rows,
        )

    def _derive_bounds_and_intervals_from_priorities(self, grid_size: int = 2000):
        derive_bounds_and_intervals(
            self.priorities_state.numerical_priorities,
            self.dataset,
            self.sample_state.sample,
            self._feature_label,
            grid_size=grid_size,
        )

    ################################################################
    # Finds element in dataset which prediction is closest to target
    ################################################################

    ########################################################
    # Filter dataset based on bounds and categorical mapping

    ################################################################
    # Finds element in dataset (1)
    ################################################################

    def find_closest_elem(self) -> None:
        """Stage 1: the priority-respecting dataset row closest to the target."""
        self.sample_state.target_exemplar = closest_row(
            self.model_pred,
            filter_rows(self.dataset, self.priorities_state),
            self.target,
            self.target_exemplar_epsilon,
        )

    ##################################
    # Calculating Shapley values (3)
    ##################################

    def calc_shapley(self, r, use_approximation: bool = False, num_samples: int = 200):
        """Stage 3: Shapley values of ``r`` against the target exemplar.

        Stores them in ``sample_state.shapley_values``; see
        :func:`shapley.shapley_values`.
        """
        phi, self.sample_state.shapley_values = shapley_values(
            self.model_pred,
            r,
            self.sample_state.target_exemplar,
            self.priorities_state.numerical_priorities,
            self.priorities_state.categorical_priorities,
            use_approximation=use_approximation,
            num_samples=num_samples,
            py_rng=self._py_rng,
        )
        return phi

    #################################################################
    # Confirm existence of solution for each categorical combination (4)
    #################################################################

    def extract_for_linear_search(self):
        """Stage 4.3: linearise the model and build one LP target per combination.

        Sets ``sample_state.shap_coeffs`` (per-unit coefficients) and returns
        ``{combination_id: target}`` for ``coefficients . x = target``.
        """
        cat_combinations = self.sample_state.all_combinations
        coefficients, coef_times_original = unit_coefficients(
            self.sample_state.shapley_values["numerical"],
            self.sample_state.sample,
            self.sample_state.target_exemplar,
            self.priorities_state.numerical_priorities,
            self.priorities_state.non_actionable_indices,
        )
        self.sample_state.shap_coeffs = coefficients

        new_targets = {}
        if len(cat_combinations[0]) != 0:
            groups = list(self.sample_state.limited_priorities["categorical"])
            for combo_idx, combo in cat_combinations.items():
                if not combo:
                    logger.debug("Empty categorical combination %s, skipping", combo_idx)
                    continue
                result = categorical_contribution(
                    combo,
                    groups,
                    self.sample_state.shapley_values["categorical"],
                    self.sample_state.sample,
                    self.sample_state.target_exemplar,
                )
                new_targets[combo_idx] = (
                    self._pass_target(combo_idx, result)
                    - result
                    - self.basic_prediction
                    + coef_times_original
                )
        else:
            new_targets[0] = self._pass_target(0, 0.0) - self.basic_prediction + coef_times_original
        return new_targets

    def _reachable_surrogate_range(self) -> tuple:
        return reachable_range(
            self.basic_prediction,
            self.sample_state.shap_coeffs,
            self._search_bounds(),
            self.sample_state.sample,
            set(self.priorities_state.non_actionable_indices),
        )

    def _pass_target(self, combo_idx: int, categorical_contribution: float) -> float:
        """Target for this pass: the effective target, clipped to what is reachable.

        When the trust region cannot span the whole gap, the pass aims at the
        closest reachable surrogate value instead of an impossible one; the
        next pass re-linearises there and continues toward the target.
        """
        wanted = self._effective_target()
        lo, hi = self._reachable_surrogate_range()
        lo += float(categorical_contribution)
        hi += float(categorical_contribution)
        clipped = min(max(wanted, lo), hi)
        if abs(clipped - wanted) > 1e-9:
            logger.info(
                "[Stage 4.3] Combo %d: target %.4f is outside the reachable "
                "surrogate range [%.4f, %.4f]; aiming at %.4f this pass.",
                combo_idx,
                wanted,
                lo,
                hi,
                clipped,
            )
        self._pass_targets[combo_idx] = float(clipped)
        return float(clipped)

    def confirm_existence_of_solution_for_combo(self):
        """Stage 4: prune combinations, linearise, and find one warm start per combination.

        Returns:
            dict: ``{combination_id: warm start}``; see
            :func:`warm_start.solve_warm_starts`.
        """
        logger.info(
            "[Stage 4.1] Pruning categorical combinations using Shapley values "
            "(zero-impact groups collapse to the sample's own category)."
        )
        self.sample_state.limited_priorities, self.sample_state.all_combinations = (
            limit_combinations(
                self.priorities_state.priorities,
                self.sample_state.shapley_values["categorical"],
                self.sample_state.sample,
                self.sample_state.target_exemplar,
            )
        )
        logger.info(
            "[Stage 4.1] %d categorical combination(s) survive after pruning.",
            len(self.sample_state.all_combinations),
        )

        logger.info(
            "[Stage 4.2] Caching baseline prediction for the original sample "
            "(used as anchor in the linear approximation)."
        )
        self.basic_prediction = self.model_pred([self.sample_state.sample])[0]
        logger.info(
            "[Stage 4.2] basic_prediction=%.4f (target=%.4f)",
            float(self.basic_prediction),
            float(self.target),
        )

        logger.info(
            "[Stage 4.3] Building per-combo LP targets and per-feature linear "
            "coefficients from Shapley values."
        )
        self._pass_targets = {}
        target_for_combo = self.extract_for_linear_search()
        logger.info("[Stage 4.3] LP targets per categorical combo: %s", target_for_combo)
        logger.debug(
            "[Stage 4.3] Per-feature shap_coeffs (unit phi): %s", self.sample_state.shap_coeffs
        )

        logger.info(
            "[Stage 4.4] Selecting actionable numerical features and gathering "
            "their bounds for the LP solver."
        )
        indices_to_modify = [
            i
            for i in self.sample_state.shap_coeffs
            if i not in self.priorities_state.non_actionable_indices
        ]
        coefficients = [self.sample_state.shap_coeffs[key] for key in indices_to_modify]
        search_bounds = self._search_bounds()
        bounds = [search_bounds[key] for key in indices_to_modify]
        logger.info(
            "[Stage 4.4] %d actionable numerical features: %s",
            len(indices_to_modify),
            indices_to_modify,
        )
        logger.debug("[Stage 4.4] LP coefficients=%s | bounds=%s", coefficients, bounds)

        warm_starts, gaps = solve_warm_starts(
            target_for_combo,
            self.sample_state.all_combinations,
            indices_to_modify,
            coefficients,
            bounds,
            self.sample_state.sample,
            self._surrogate,
            lambda combination_id: self._pass_targets.get(combination_id, self._effective_target()),
            self.model_pred,
            self.target,
            self.epsilon,
        )

        if self._warm_start_info is None:
            self._warm_start_info = warm_start_health(target_for_combo, warm_starts, gaps)
            logger.info(
                "[Stage 4.6] Warm-start health: %d/%d combos feasible; best "
                "warm-start model |gap|=%s | linear |gap|=%s.",
                self._warm_start_info["feasible_combos"],
                self._warm_start_info["total_combos"],
                self._warm_start_info["best_warmstart_model_gap"],
                self._warm_start_info["best_warmstart_linear_gap"],
            )

        return warm_starts

    #################################################################
    # Functions for optimisation
    #################################################################

    def constraint_function(
        self, x, shap_dict, sample, target_exemplar, priorities_for_search, basic_prediction
    ):
        """Surrogate prediction ``h(x)``; see :func:`surrogate.surrogate_prediction`."""
        return surrogate_prediction(
            x,
            self.sample_state.shap_coeffs,
            shap_dict,
            sample,
            target_exemplar,
            priorities_for_search,
            basic_prediction,
        )

    def _surrogate(self, x):
        """``h(x)`` linearised at the current point."""
        return self.constraint_function(
            x,
            self.sample_state.shapley_values,
            self.sample_state.sample,
            self.sample_state.target_exemplar,
            self.sample_state.limited_priorities,
            basic_prediction=self.basic_prediction,
        )

    def calculate_total_weight(self, values):
        """Priority benefit of ``values``; see :func:`scoring.total_weight`."""
        return total_weight(
            values,
            self.priorities_state.numerical_priorities,
            self.priorities_state.categorical_priorities,
        )

    #################################################################
    # Main function to find counterfactuals
    #################################################################

    def _stage1_find_exemplar(self):
        """Locate the anchor exemplar, with one fallback.

        Selection strategy, tried in order (the first that succeeds wins):

          1. ``dataset_priority_filtered`` -- original behaviour: the
             priority-feasible dataset row whose prediction is closest to the
             target (:meth:`find_closest_elem`).
          2. ``random_in_range`` -- if (1) finds no feasible row, randomly
             generate an exemplar by sampling each actionable feature's allowed
             range and taking the first point within ``epsilon`` of the target.
             This is **not** the random-search comparison method: preference /
             priority score is deliberately ignored here; we only need any
             reachable, in-range anchor for MINLP to then optimise for the
             highest preference score.

        Sets ``self.sample_state.target_exemplar``, records the chosen strategy
        on ``self.exemplar_source`` and the anchor's ``|pred - target|`` on
        ``self.exemplar_pred_distance``. Runs once per ``find_counterfactuals``
        call; the exemplar is the fixed anchor every refinement iteration
        linearises against.
        """
        logger.info("--- STAGE 1/6: Locate target exemplar (with fallbacks) ---")
        logger.info(
            "Goal: pick an anchor whose model prediction is close to the "
            "requested target. It anchors the linear (Shapley) approximation "
            "used in later stages."
        )

        try:
            self.find_closest_elem()
            self.exemplar_source = "dataset_priority_filtered"
            self.exemplar_pred_distance = self._exemplar_pred_distance()
            logger.info(
                "Target exemplar via '%s': |pred - target|=%.4f (target=%.4f).",
                self.exemplar_source,
                self.exemplar_pred_distance,
                float(self.target),
            )
            return
        except Exception as primary_exc:
            logger.warning(
                "[Stage 1] Primary exemplar selection ('dataset_priority_filtered') failed: %s",
                primary_exc,
            )
            try:
                self._generate_exemplar_in_range(
                    max_iterations=int(getattr(self, "_fallback_random_max_iterations", 10000)),
                    random_seed=getattr(self, "_random_seed", None),
                )
                self.exemplar_source = "random_in_range"
                self.exemplar_pred_distance = self._exemplar_pred_distance()
                logger.warning(
                    "[Stage 1] Randomly generated in-range exemplar (|pred - target|=%.4f).",
                    self.exemplar_pred_distance,
                )
                return
            except Exception as rnd_exc:
                self.exemplar_source = "none"
                logger.error(
                    "[Stage 1] Could not obtain any exemplar (dataset or random in-range): %s",
                    rnd_exc,
                )
                raise

    def _exemplar_pred_distance(self) -> float:
        """``|model_pred(target_exemplar) - target|`` for the current exemplar."""
        pred = float(
            np.asarray(self.model_pred([self.sample_state.target_exemplar])).reshape(-1)[0]
        )
        return abs(pred - float(self.target))

    def _generate_exemplar_in_range(self, max_iterations: int = 10000, random_seed=None) -> float:
        """Stage 1 fallback: a random in-range anchor; returns its distance to the target."""
        if random_seed is not None:
            self._np_rng = make_rngs(random_seed)[0]
        self.sample_state.target_exemplar, distance = generate_in_range(
            self.model_pred,
            self.sample_state.sample,
            self.priorities_state,
            self.target,
            self.epsilon,
            max_iterations,
            self._np_rng,
        )
        return distance

    def probe_allowed_region_feasibility(
        self, max_iterations: int = 5000, random_seed=None
    ) -> dict:
        """Measure whether the allowed region can reach the target at all.

        Randomly samples the allowed region (ignoring priorities) and reports
        whether any point lands within ``epsilon`` of the target, and the
        smallest ``|pred - target|`` seen. This separates *feasibility*
        failures (the region cannot reach the target) from *convergence*
        failures (it can, but the optimiser did not). Runs Stage 0 first.

        Returns:
            dict: ``feasible_within_epsilon``, ``min_pred_distance`` and
            ``iterations_run``.
        """
        self._derive_bounds_and_intervals_from_priorities()
        if random_seed is not None:
            self._np_rng = make_rngs(random_seed)[0]
        return probe_feasibility(
            self.model_pred,
            self.sample_state.sample,
            self.priorities_state,
            self.target,
            self.epsilon,
            max_iterations,
            self._np_rng,
        )

    def _stage2_log_bounds(self):
        """Surface the numerical bounds used by both LP and SLSQP."""
        logger.info("--- STAGE 2/6: Collect numerical-feature bounds from priorities ---")
        logger.info(
            "These (min, max) pairs constrain how far each numerical feature is "
            "allowed to move during the LP and SLSQP searches."
        )
        logger.info("Bounds for numerical features: %s", self.priorities_state.bounds)
        logger.info(
            "Non-actionable feature indices (frozen at sample values): %s",
            self.priorities_state.non_actionable_indices,
        )

    def _effective_target(self) -> float:
        """Target the *surrogate* must hit so the true model lands on ``target``.

        Equals ``target`` unless a previous pass measured a residual
        ``f(x) - h(x)`` that the additive correction is compensating for.
        """
        return float(self.target) - float(self._target_correction)

    def _search_bounds(self) -> dict[int, tuple]:
        """Per-feature ``(lo, hi)`` for the LP and SLSQP, trust region included."""
        return self._trust_region.limit(self.priorities_state.bounds, self.sample_state.sample)

    def _run_one_pass(self, shap_approx, num_samples):
        """Execute stages 3, 4 and 5 once for the current ``sample_state.sample``.

        Returns the list of candidate counterfactuals (one per surviving
        categorical combination). Stage 6 selection is delegated to
        :meth:`_select_best_candidate` so the iterative loop in
        :meth:`find_counterfactuals` can reuse both helpers cleanly.
        """
        # Stage 3: Shapley values for the (possibly updated) current sample.
        logger.info("--- STAGE 3/6: Compute Shapley values (sample vs. target exemplar) ---")
        logger.info(
            "Shapley values quantify each feature's contribution to the gap "
            "between the prediction on the sample and on the target exemplar. "
            "Numerical features get one value each; one-hot categorical groups "
            "are consolidated into a single value per group."
        )
        current_sample = [float(v) for v in np.asarray(self.sample_state.sample, dtype=float)]
        if (
            self.sample_state.shapley_values is not None
            and getattr(self, "_last_shapley_sample", None) == current_sample
        ):
            logger.info(
                "[Stage 3] Iterate unchanged since the last pass: reusing the "
                "cached Shapley values (saves a full re-linearisation)."
            )
        else:
            self.calc_shapley(
                self.sample_state.sample, use_approximation=shap_approx, num_samples=num_samples
            )
            self._last_shapley_sample = current_sample
        logger.info(
            "Shapley values (numerical): %s", self.sample_state.shapley_values.get("numerical")
        )
        logger.info(
            "Shapley values (categorical groups): %s",
            self.sample_state.shapley_values.get("categorical"),
        )

        # Stage 4: warm-start LP per surviving categorical combination.
        logger.info("--- STAGE 4/6: Build initial feasible solutions per categorical combo ---")
        logger.info(
            "For each surviving categorical combination, solve a linear program "
            "(in the Shapley-linearised model) to find numerical values that "
            "land within +/- epsilon of the target. These become warm starts "
            "for the nonlinear SLSQP optimisation in stage 5."
        )
        # An infeasible LP under a tight trust region means the region cannot
        # reach the target, not that the problem is infeasible: enlarge and
        # retry, reusing the Shapley values just computed (no model calls).
        while True:
            init_vals_per_combo = self.confirm_existence_of_solution_for_combo()
            if init_vals_per_combo or not self._trust_region.grow():
                break
            logger.info(
                "[Stage 4] No warm start inside the trust region; retrying "
                "with the enlarged region."
            )
        logger.info("Initial values per combination: %s", init_vals_per_combo)
        logger.info("Limited priorities used for search: %s", self.sample_state.limited_priorities)

        # Stage 5: SLSQP per combo. Wrappers and bounds match the original flow.
        logger.info("--- STAGE 5/6: SLSQP optimisation per categorical combination ---")
        logger.info(
            "Minimise the priority cost subject to the Shapley-linear constraint "
            "h(x) in [target-epsilon, target+epsilon]. One run per categorical "
            "combination; without categorical features the loop runs once."
        )
        bounds = self._search_bounds()
        numerical_indices = list(self.priorities_state.numerical_priorities.keys())
        counterfactuals = []
        for i, values in init_vals_per_combo.items():
            label = f"[combo {i + 1}/{len(init_vals_per_combo)}]"
            logger.info(
                "%s Preparing input from initial LP solution and categorical assignment...", label
            )
            prepared_input = self.sample_state.sample.copy()
            for idx, val in values["categorical_combo"].items():
                prepared_input[idx] = val
            for idx, val in values["initial_solution"].items():
                prepared_input[idx] = val
            logger.debug("Prepared input: %s", prepared_input)

            # The real target, unless the trust region can only reach part of
            # the way (then the closest reachable value).
            pass_target = self._pass_targets.get(i, self._effective_target())
            result, complete_solution, h_solution = optimise_combination(
                prepared_input,
                numerical_indices,
                self.priorities_state.numerical_priorities,
                bounds,
                self._surrogate,
                self.calculate_total_weight,
                pass_target,
                self.epsilon,
                label=label,
            )
            counterfactuals.append(complete_solution)
            model_prediction = float(self.model_pred([complete_solution])[0])
            abs_gap = abs(model_prediction - float(self.target))
            logger.info(
                "%s SLSQP done: success=%s | iterations=%s | model_pred(cf)=%.4f | "
                "constraint h(x)=%.4f | objective(weight)=%.4f",
                label,
                getattr(result, "success", None),
                getattr(result, "nit", "?"),
                model_prediction,
                float(h_solution),
                float(-result.fun),
            )
            iter_no = getattr(self, "_workflow_iteration", None)
            if iter_no is None:
                self._workflow_log(
                    "4) Proposition combo=%d: prediction=%.6f | target=%.6f | |gap|=%.6f",
                    i + 1,
                    model_prediction,
                    float(self.target),
                    abs_gap,
                )
            else:
                self._workflow_log(
                    "4) Iteration %d proposition combo=%d: prediction=%.6f | target=%.6f | |gap|=%.6f",
                    int(iter_no),
                    i + 1,
                    model_prediction,
                    float(self.target),
                    abs_gap,
                )
            logger.debug(
                "%s Optimal x=%s | message=%s", label, result.x, getattr(result, "message", "")
            )

        return counterfactuals

    def _priority_benefit(self, cf):
        return priority_benefit(
            cf,
            self.priorities_state.numerical_priorities,
            self.priorities_state.categorical_priorities,
        )

    def _feature_priority(self, idx: int, value: float) -> float:
        return feature_priority(self.priorities_state.numerical_priorities.get(idx, {}), value)

    def _priority_peaks(self) -> dict[int, tuple]:
        """Cached :func:`scoring.priority_peaks`; priorities do not change during a run."""
        if getattr(self, "_priority_peaks_cache", None) is None:
            self._priority_peaks_cache = priority_peaks(self.priorities_state.numerical_priorities)
        return self._priority_peaks_cache

    def _clip_to_allowed(self, idx: int, value: float) -> float:
        return clip_to_allowed(
            value,
            self.priorities_state.numerical_priorities.get(idx, {}),
            self.priorities_state.bounds.get(idx, (None, None)),
        )

    def _refinement_context(self) -> RefinementContext:
        """Snapshot of the current search state for the refinement steps."""
        return RefinementContext(
            model=self.model_pred,
            target=self.target,
            epsilon=self.epsilon,
            coefficients=getattr(self.sample_state, "shap_coeffs", None),
            immutable=set(self.priorities_state.non_actionable_indices),
            surrogate=self._surrogate,
            priority_benefit=self._priority_benefit,
            feature_priority=self._feature_priority,
            clip=self._clip_to_allowed,
            priority_peaks=self._priority_peaks,
            feature_label=self._feature_label,
        )

    def find_counterfactuals(
        self,
        shap_approx=False,
        num_samples=200,
        max_iterations=10,
        patience=5,
        return_when_fails=True,
        fallback_random_max_iterations=10000,
        peak_lock_in=True,
        peak_lock_in_max_sweeps=3,
        random_seed=None,
        trust_region=True,
        trust_region_init=0.25,
        trust_region_min=0.02,
        trust_region_max=1.0,
        residual_correction=True,
        restoration=True,
    ):
        """Find a counterfactual via iterative Shapley re-linearisation.

        Stages 1 and 2 (locate target exemplar, gather bounds) run once.
        Stages 3-6 (Shapley, LP warm starts, SLSQP, candidate selection)
        run inside a refinement loop: after each pass we evaluate the
        chosen candidate with the real model and advance the working sample
        to it, re-linearising against the same exemplar.

        Two incumbents are tracked, so that the priority benefit (the
        quantity the optimisation claims to maximise) is what decides the
        returned counterfactual:

          * ``best_feasible``: among candidates inside the target band
            (``|model_pred - target| <= epsilon``), the one with the highest
            priority benefit. Returned whenever it exists.
          * ``best_infeasible``: only used while no candidate has entered the
            band; ranked by ``|model_pred - target|``.

        The loop therefore does *not* stop when the band is first reached: the
        band becomes a hard constraint and the search keeps trying to improve
        the priority benefit inside it.

        The loop stops when:
          * ``max_iterations`` passes have been executed, or
          * ``patience`` consecutive iterations have failed to improve the
            active incumbent (priority benefit once inside the band, distance
            to target before that), or
          * an internal pass raised an exception (e.g. infeasible LP).

        Args:
            shap_approx: If True, use Monte-Carlo Shapley estimation.
            num_samples: Number of subsets used by the approximate
                Shapley estimator.
            max_iterations: Maximum number of refinement passes.
            patience: Stop after this many consecutive iterations without
                improving the active incumbent.
            return_when_fails: If True (default) return the best candidate
                found even when the target was never reached, with a
                warning log and full status on
                ``self.last_search_result``. If False, return ``None``
                when the target was not reached.
            peak_lock_in: If True (default) run the Stage 7 peak lock-in
                sweep (:meth:`_peak_lock_in`) on the feasible incumbent
                before returning.
            peak_lock_in_max_sweeps: Maximum number of Stage 7 sweeps.
            random_seed: Seeds the stochastic steps of the search (the
                Stage 1 fallback anchor and approximate Shapley subset
                sampling). Pass an int for a reproducible run; the global
                NumPy and ``random`` state is left untouched. With ``None``
                the global random streams are used.
            trust_region: If True (default) each pass may only move a
                feature by a fraction of its allowed span, so SLSQP stays
                where the Shapley-linear surrogate is accurate. The radius
                grows after an accepted step and shrinks after a rejected
                one; a pass whose warm-start LP is infeasible inside the
                region enlarges it and retries without new model calls.
            trust_region_init/min/max: Radius bounds as a fraction of each
                feature's allowed span.
            residual_correction: If True (default) a rejected candidate's
                measured residual ``f(x) - h(x)`` is subtracted from the
                target the surrogate aims at, so the next pass asks the
                linear model for the change the true model actually needs.
                Costs no extra model calls.
            restoration: If True (default) a candidate that lands just
                outside the band is pulled back in with a least-norm
                Shapley-Newton correction (:meth:`_restore_to_band`)
                instead of being discarded. Costs up to 3 model calls per
                pass and keeps most of the candidate's priority.

        Returns:
            list | None: The selected counterfactual feature vector, or
            ``None`` when the target was not reached and
            ``return_when_fails`` is False.

        Side effects:
            Sets ``self.last_search_result`` with keys ``reached_target``,
            ``distance``, ``iterations_run``, ``stop_reason``, ``best_cf``,
            ``history``, ``cf_source`` (``anchor`` when the returned CF is the
            Stage 1 exemplar, ``optimiser`` when it came out of SLSQP),
            ``priority_score``, ``anchor_priority_score`` and
            ``priority_gain_vs_anchor`` (0.0 when the anchor wins).
        """
        logger.info("=" * 78)
        logger.info(
            "MINLP COUNTERFACTUAL SEARCH | target=%.4f | epsilon=%.4f | "
            "max_iterations=%d | patience=%d | shap_approx=%s | "
            "num_samples=%d | return_when_fails=%s | trust_region=%s | "
            "residual_correction=%s",
            float(self.target),
            float(self.epsilon),
            int(max_iterations),
            int(patience),
            bool(shap_approx),
            int(num_samples),
            bool(return_when_fails),
            bool(trust_region),
            bool(residual_correction),
        )
        logger.info("=" * 78)

        self.exemplar_source = None
        self.exemplar_pred_distance = None
        self._warm_start_info = None
        self._last_search_exception = None
        self._priority_peaks_cache = None
        self._fallback_random_max_iterations = int(fallback_random_max_iterations)
        self._random_seed = None if random_seed is None else int(random_seed)
        self._np_rng, self._py_rng = make_rngs(self._random_seed)
        self._trust_region = TrustRegion(
            fraction=float(trust_region_init) if trust_region else None,
            minimum=float(trust_region_min),
            maximum=float(trust_region_max),
        )
        self._target_correction = 0.0
        self._residual_correction_enabled = bool(residual_correction)
        self._restoration_enabled = bool(restoration)
        self._last_shapley_sample = None
        self._pass_targets = {}

        # Stage 0: infer bounds directly from priority functions.
        self._derive_bounds_and_intervals_from_priorities()

        # Stages 1 and 2 are anchor stages: they only depend on the
        # original sample/exemplar/priorities, so we run them exactly once.
        self._stage1_find_exemplar()
        self._stage2_log_bounds()
        self._log_workflow_initial_and_bounds()

        original_sample = list(self.sample_state.sample)
        # Keep the selected exemplar as a valid fallback when it already lies
        # within the final target band. This matters for generated in-range
        # anchors: if later surrogate optimisation fails or drifts, returning a
        # valid in-range anchor is better than reporting no CF.
        anchor_cf = list(np.asarray(self.sample_state.target_exemplar, dtype=float))
        anchor_pred = float(np.asarray(self.model_pred([anchor_cf])).reshape(-1)[0])
        anchor_distance = abs(anchor_pred - float(self.target))
        anchor_priority = self._priority_benefit(anchor_cf)

        best_feasible = None
        best_feasible_priority = float("-inf")
        best_feasible_distance = float("inf")
        best_feasible_source = None
        best_infeasible = None
        best_infeasible_distance = float("inf")

        if anchor_distance <= float(self.epsilon):
            best_feasible = anchor_cf
            best_feasible_priority = (
                anchor_priority if anchor_priority is not None else float("-inf")
            )
            best_feasible_distance = anchor_distance
            best_feasible_source = "anchor"
            logger.info(
                "Initial exemplar is already a valid fallback CF: "
                "model_pred=%.4f | distance=%.4f | priority=%s. It is only "
                "kept as the incumbent until an SLSQP candidate beats its "
                "priority benefit.",
                anchor_pred,
                anchor_distance,
                anchor_priority,
            )
        no_progress = 0
        history = []
        stop_reason = "max_iterations"
        restoration_hits = 0

        for iteration in range(int(max_iterations)):
            logger.info(
                "############ REFINEMENT ITERATION %d/%d ############",
                iteration + 1,
                int(max_iterations),
            )
            self._workflow_iteration = iteration + 1
            try:
                candidates = self._run_one_pass(shap_approx, num_samples)
                cf = select_best_candidate(self._refinement_context(), candidates)
            except Exception as exc:
                logger.warning("Iteration %d failed during search pass: %s", iteration + 1, exc)
                stop_reason = "search_failed"
                self._last_search_exception = str(exc)
                break

            eval_info = evaluate_candidate(self._refinement_context(), cf)
            if not eval_info["feasible"] and self._restoration_enabled:
                restored_cf, restored_eval = restore_to_band(
                    self._refinement_context(), cf, eval_info
                )
                if restored_cf is not None:
                    cf, eval_info = restored_cf, restored_eval
                    restoration_hits += 1
            priority = eval_info["priority"]
            improved = False
            # Surrogate agreement over this step: both models share the value
            # ``basic_prediction`` at the current iterate, so no extra model
            # call is needed to compare their predicted changes.
            anchor_value = float(self.basic_prediction)
            surrogate_change = float(eval_info["h_x"]) - anchor_value
            model_change = float(eval_info["model_pred"]) - anchor_value
            rho = model_change / surrogate_change if abs(surrogate_change) > 1e-12 else float("nan")
            residual = float(eval_info["model_pred"]) - float(eval_info["h_x"])
            # A step is worth taking when it lands in the band or simply gets
            # closer to the target than the point it started from.
            iterate_distance = abs(anchor_value - float(self.target))
            accepted = bool(
                eval_info["feasible"] or eval_info["distance"] < iterate_distance - 1e-12
            )

            if eval_info["feasible"]:
                # Inside the band: rank on the priority benefit only.
                if best_feasible is None or (
                    priority is not None and priority > best_feasible_priority
                ):
                    best_feasible = list(cf)
                    best_feasible_priority = priority if priority is not None else float("-inf")
                    best_feasible_distance = eval_info["distance"]
                    best_feasible_source = "optimiser"
                    improved = True
            elif best_feasible is None and eval_info["distance"] < best_infeasible_distance:
                # No candidate inside the band yet: rank on distance to target.
                best_infeasible = list(cf)
                best_infeasible_distance = eval_info["distance"]
                improved = True

            best_priority_so_far = (
                f"{best_feasible_priority:.4f}"
                if best_feasible_priority != float("-inf")
                else "n/a"
            )
            best_distance_so_far = (
                f"{best_feasible_distance:.4f}"
                if best_feasible is not None
                else (
                    f"{best_infeasible_distance:.4f}"
                    if best_infeasible_distance != float("inf")
                    else "n/a"
                )
            )
            logger.info(
                "[Iter %d] model_pred(cf)=%.4f | h(x)=%.4f | distance=%.4f | "
                "feasible=%s | priority=%s | best_priority=%s | "
                "best_distance=%s | improved_vs_best=%s | rho=%.3f | "
                "residual=%.4f | trust=%s",
                iteration + 1,
                eval_info["model_pred"],
                eval_info["h_x"],
                eval_info["distance"],
                eval_info["feasible"],
                f"{priority:.4f}" if priority is not None else "n/a",
                best_priority_so_far,
                best_distance_so_far,
                improved,
                rho,
                residual,
                f"{self._trust_region.fraction:.3f}"
                if self._trust_region.fraction is not None
                else "off",
            )

            if accepted:
                self._target_correction = 0.0
                if (
                    self._trust_region.fraction is not None
                    and np.isfinite(rho)
                    and 0.5 <= rho <= 1.5
                ):
                    self._trust_region.grow()
            else:
                if self._residual_correction_enabled:
                    self._target_correction = residual
                    logger.info(
                        "[Iter %d] Residual correction: surrogate now aims at "
                        "%.4f so the model lands on %.4f.",
                        iteration + 1,
                        self._effective_target(),
                        float(self.target),
                    )
                self._trust_region.shrink()

            # An accepted step is progress even when it does not yet beat the
            # incumbent: the trust-region walk needs several passes to cross
            # the gap, and patience must not cut it short.
            if improved or accepted:
                no_progress = 0
            else:
                no_progress += 1
                logger.info(
                    "[Iter %d] No improvement vs incumbent (%d/%d).",
                    iteration + 1,
                    no_progress,
                    int(patience),
                )

            history.append(
                {
                    "iteration": iteration + 1,
                    "model_pred": eval_info["model_pred"],
                    "h_x": eval_info["h_x"],
                    "distance": eval_info["distance"],
                    "feasible": eval_info["feasible"],
                    "priority": priority,
                    "improved_vs_best": improved,
                    "accepted_step": accepted,
                    "rho": (float(rho) if np.isfinite(rho) else None),
                    "residual": residual,
                    "trust_fraction": self._trust_region.fraction,
                    "target_correction": self._target_correction,
                }
            )

            if no_progress >= int(patience):
                logger.info(
                    "Stopping: %d consecutive iterations without improving the %s incumbent.",
                    int(patience),
                    "priority" if best_feasible is not None else "distance",
                )
                stop_reason = (
                    "priority_stagnation" if best_feasible is not None else "patience_exhausted"
                )
                break

            # Advance only on an accepted (in-band) step, so the surrogate is
            # always re-linearised at a point the true model agrees with. With
            # both step controls off, keep the historical behaviour of moving
            # to every candidate.
            step_control = (
                self._trust_region.fraction is not None
            ) or self._residual_correction_enabled
            if accepted or not step_control:
                self.sample_state.sample = list(cf)
                logger.info(
                    "[Iter %d] Advancing: next iteration's sample = current CF.", iteration + 1
                )
            else:
                logger.info(
                    "[Iter %d] Step rejected (|f - target|=%.4f > epsilon=%.4f); "
                    "keeping the current iterate and retrying with a smaller "
                    "step / corrected target.",
                    iteration + 1,
                    eval_info["distance"],
                    float(self.epsilon),
                )

        # Restore the original sample so explainer state stays consistent
        # for any caller that inspects sample_state after the search.
        self.sample_state.sample = original_sample
        self._workflow_iteration = None

        peak_info = None
        if peak_lock_in and best_feasible is not None:
            improved_cf, peak_info = lock_in_priority_peaks(
                self._refinement_context(), best_feasible, max_sweeps=int(peak_lock_in_max_sweeps)
            )
            if peak_info["accepted_moves"] > 0:
                best_feasible = improved_cf
                best_feasible_priority = float(peak_info["priority_after"])
                best_feasible_distance = abs(
                    float(peak_info["final_prediction"]) - float(self.target)
                )
                best_feasible_source = f"{best_feasible_source}+peak_lock_in"

        reached = best_feasible is not None
        if reached:
            best_cf = best_feasible
            best_distance = best_feasible_distance
            best_priority = (
                best_feasible_priority if best_feasible_priority != float("-inf") else None
            )
            cf_source = best_feasible_source
            if stop_reason == "max_iterations":
                stop_reason = "target_reached"
        else:
            best_cf = best_infeasible
            best_distance = best_infeasible_distance
            best_priority = self._priority_benefit(best_cf) if best_cf is not None else None
            cf_source = "optimiser" if best_cf is not None else None

        # The anchor is a priority-agnostic in-range point, so a run that
        # returns it must not be credited with a priority gain.
        if cf_source == "anchor" or best_priority is None or anchor_priority is None:
            priority_gain_vs_anchor = 0.0 if cf_source == "anchor" else None
        else:
            priority_gain_vs_anchor = float(best_priority) - float(anchor_priority)

        self.last_search_result = {
            "reached_target": reached,
            "distance": best_distance,
            "iterations_run": len(history),
            "stop_reason": stop_reason,
            "best_cf": list(best_cf) if best_cf is not None else None,
            "history": history,
            "exemplar_source": self.exemplar_source,
            "exemplar_pred_distance": self.exemplar_pred_distance,
            "warm_start": self._warm_start_info,
            "search_exception": self._last_search_exception,
            "cf_source": cf_source,
            "priority_score": best_priority,
            "anchor_priority_score": anchor_priority,
            "anchor_distance": anchor_distance,
            "priority_gain_vs_anchor": priority_gain_vs_anchor,
            "peak_lock_in": peak_info,
            "step_control": {
                "trust_region_enabled": bool(trust_region),
                "trust_fraction_final": self._trust_region.fraction,
                "residual_correction_enabled": self._residual_correction_enabled,
                "accepted_steps": sum(1 for h in history if h.get("accepted_step")),
                "rejected_steps": sum(1 for h in history if h.get("accepted_step") is False),
                "restoration_hits": int(restoration_hits),
            },
        }

        logger.info("=" * 78)
        logger.info(
            "MINLP SEARCH DONE | reached_target=%s | stop_reason=%s | "
            "iterations=%d | best_distance=%.4f | cf_source=%s | "
            "priority=%s | anchor_priority=%s | gain_vs_anchor=%s",
            reached,
            stop_reason,
            len(history),
            best_distance if best_distance != float("inf") else float("nan"),
            cf_source,
            best_priority,
            anchor_priority,
            priority_gain_vs_anchor,
        )
        if cf_source == "anchor":
            logger.warning(
                "Returned CF is the Stage 1 anchor exemplar (source=%s): no "
                "SLSQP candidate beat its priority benefit (%s). This run "
                "carries no optimisation gain.",
                self.exemplar_source,
                anchor_priority,
            )
        logger.info("=" * 78)

        if reached:
            return best_cf
        if return_when_fails:
            if best_cf is None:
                logger.warning(
                    "No candidate was produced; returning None despite return_when_fails=True."
                )
                return None
            logger.warning(
                "Returning best CF although it did NOT reach target "
                "(distance=%.4f > epsilon=%.4f). See "
                "self.last_search_result for full status.",
                best_distance,
                float(self.epsilon),
            )
            return best_cf
        logger.warning("Target not reached and return_when_fails=False; returning None.")
        return None

    def find_counterfactuals_for_binary(
        self,
        target_class=1,
        threshold=0.5,
        expected_counterfactuals=10,
        max_iterations=100,
        shap_approx=False,
        num_samples=200,
        return_top_n=None,
    ):
        """
        Find counterfactual explanations for binary classification problems.

        This method extends the MINLP approach to work with binary classifiers by searching
        for samples that cross the decision threshold in the desired direction. Instead of
        finding an exact target prediction value, it optimizes within a target range that
        ensures the prediction crosses the threshold appropriately.

        The algorithm:
        1. Sets target prediction to 0.75 (for class 1) or 0.25 (for class 0)
        2. Uses confidence interval to create an acceptable range around the target
        3. Runs MINLP optimization for each categorical combination
        4. Returns top-N counterfactuals ranked by priority weight

        Args:
            target_class: Desired class (0 or 1)
            threshold: Decision threshold for binary classification (default: 0.5)
            expected_counterfactuals: Number of counterfactuals to find
            max_iterations: Maximum optimization iterations per combination
            shap_approx: If True, uses sampling-based Shapley approximation
            num_samples: Number of samples for Shapley approximation
            return_top_n: If specified, return only top N most preferable counterfactuals
                         (default: return all found)

        Returns:
            tuple: (counterfactuals, predictions, scores, found_counts)
                - counterfactuals: List of counterfactual samples
                - predictions: List of model predictions for each counterfactual
                - scores: List of preference scores based on priority weights
                - found_counts: List indicating iteration number when each CF was found

        Raises:
            ValueError: If no samples found satisfy the classification constraint
        """
        logger.info("Starting MINLP-based binary classification counterfactual search...")
        logger.info(f"Target class: {target_class}, Threshold: {threshold}")

        # Set target prediction value based on desired class
        # For class 1: target ~0.75 (well above threshold)
        # For class 0: target ~0.25 (well below threshold)
        if target_class == 1:
            self.target = 0.75
        else:
            self.target = 0.25

        # Set confidence interval (epsilon) to ensure we're comfortably across the threshold
        self.epsilon = max(0.1, abs(self.target - threshold))

        logger.info(f"Set internal target to {self.target} with epsilon {self.epsilon}")
        self._derive_bounds_and_intervals_from_priorities()

        # Run the standard MINLP pipeline
        try:
            ############
            # 1. Finding the target exemplar in the dataset
            self.find_closest_elem()
            logger.info(
                f"Target exemplar prediction: {self.model_pred([self.sample_state.target_exemplar])[0]:.4f}"
            )
        except ValueError as e:
            logger.warning(f"Could not find target exemplar: {e}")
            logger.warning("Proceeding without target exemplar - using current sample as reference")
            self.sample_state.target_exemplar = self.sample_state.sample.copy()

        ############
        # 2. Determining bounds for numerical features
        bounds = self.priorities_state.bounds
        logger.info(f"Bounds for numerical features: {bounds}")

        ###########
        # 3. Calculate SHAP values
        self.calc_shapley(
            self.sample_state.sample, use_approximation=shap_approx, num_samples=num_samples
        )
        logger.info(
            f"SHAP values computed: {len(self.sample_state.shapley_values['numerical'])} numerical, "
            f"{len(self.sample_state.shapley_values['categorical'])} categorical features"
        )

        ###########
        # 4. Create initial values per categorical combination
        try:
            init_vals_per_combo = self.confirm_existence_of_solution_for_combo()
        except (ValueError, AssertionError) as e:
            logger.warning(f"Could not confirm solution existence: {e}")
            logger.info("Attempting to proceed with optimizations anyway...")
            init_vals_per_combo = {}

        if not init_vals_per_combo:
            logger.info(
                "No feasible combinations found. Generating initial values from sample/target..."
            )
            init_vals_per_combo = {0: {"initial_solution": {}, "categorical_combo": {}}}

        logger.info(f"Processing {len(init_vals_per_combo)} categorical combinations")

        ############
        # 5. For each categorical combination, perform MINLP optimization
        logger.info("Starting MINLP search for binary classification counterfactuals...")
        counterfactuals = []
        predictions = []
        scores = []
        found_counts = []

        for combo_id, values in init_vals_per_combo.items():
            logger.info(f"Processing categorical combination {combo_id}...")
            prepared_input = self.sample_state.sample.copy()

            if values.get("initial_solution"):
                initial_numerical = values["initial_solution"]
                cat_combo = values.get("categorical_combo", {})
            else:
                # Use sample as initial solution if none provided
                initial_numerical = {}
                cat_combo = {}

            for idx, val in cat_combo.items():
                prepared_input[idx] = val
            for idx, val in initial_numerical.items():
                prepared_input[idx] = val

            # Define constraint and objective functions
            numerical_indices = list(self.priorities_state.numerical_priorities.keys())

            def objective_wrapper(
                x, prepared_input=prepared_input, numerical_indices=numerical_indices
            ):
                temp_input = prepared_input.copy()
                for i, idx in enumerate(numerical_indices):
                    if i < len(x):
                        temp_input[idx] = x[i]
                try:
                    return self.calculate_total_weight(temp_input)
                except ValueError:
                    return -1e12

            # For binary classification, use direct model prediction instead of Shapley approximation
            def classification_constraint(
                x, prepared_input=prepared_input, numerical_indices=numerical_indices
            ):
                temp_input = prepared_input.copy()
                for i, idx in enumerate(numerical_indices):
                    if i < len(x):
                        temp_input[idx] = x[i]
                pred = self.model_pred([temp_input])[0]
                if target_class == 1:
                    return pred - threshold  # pred > threshold
                else:
                    return threshold - pred  # pred < threshold

            constraints = [{"type": "ineq", "fun": classification_constraint}]

            x0 = np.array([prepared_input[idx] for idx in numerical_indices])
            bounds_list = [bounds[idx] for idx in numerical_indices]
            for j, feature_idx in enumerate(numerical_indices):
                cfg = self.priorities_state.numerical_priorities.get(feature_idx, {})
                allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None
                if allowed_intervals:
                    x0[j] = project_to_allowed_intervals(float(x0[j]), allowed_intervals)
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": (
                                lambda x, _j=j, _intervals=allowed_intervals: (
                                    allowed_interval_constraint_value(float(x[_j]), _intervals)
                                )
                            ),
                        }
                    )

            logger.info(f"Starting optimization for combination {combo_id}...")
            logger.debug(f"Initial values: {x0}")
            logger.debug(f"Bounds: {bounds_list}")

            try:
                result = minimize(
                    lambda x: -objective_wrapper(x),  # Maximize weight (minimize negative weight)
                    x0,
                    method="SLSQP",
                    bounds=bounds_list,
                    constraints=constraints,
                    options={"maxiter": max_iterations, "ftol": 1e-6},
                )

                if (
                    result.success or not result.success
                ):  # Accept both success and partial convergence
                    complete_solution = prepared_input.copy()
                    for i, idx in enumerate(numerical_indices):
                        complete_solution[idx] = result.x[i]

                    # Verify the solution meets the classification constraint
                    pred = self.model_pred([complete_solution])[0]
                    is_valid = (pred > threshold) if target_class == 1 else (pred < threshold)

                    logger.debug(
                        f"Solution prediction: {pred:.4f}, valid: {is_valid}, message: {result.message}"
                    )

                    if is_valid:
                        counterfactuals.append(complete_solution)
                        predictions.append(pred)
                        scores.append(objective_wrapper(result.x))
                        found_counts.append(combo_id + 1)
                        logger.info(
                            f"Found valid counterfactual: pred={pred:.4f}, weight={scores[-1]:.4f}"
                        )
                    else:
                        logger.debug(
                            f"Solution doesn't meet classification constraint: pred={pred:.4f}, threshold={threshold}"
                        )

            except Exception as e:
                logger.warning(f"Error during optimization for combination {combo_id}: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        # Sort by preference score (ascending = lower cost/weight is better)
        if counterfactuals:
            sorted_indices = np.argsort(scores)
            counterfactuals = [counterfactuals[i] for i in sorted_indices]
            predictions = [predictions[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            found_counts = [found_counts[i] for i in sorted_indices]

            # Limit to top N if requested
            if return_top_n is not None and len(counterfactuals) > return_top_n:
                counterfactuals = counterfactuals[:return_top_n]
                predictions = predictions[:return_top_n]
                scores = scores[:return_top_n]
                found_counts = found_counts[:return_top_n]

            logger.info(f"Found {len(counterfactuals)} valid counterfactuals")
        else:
            logger.warning("No valid counterfactuals found!")

        return counterfactuals, predictions, scores, found_counts

    #################################################################
    # Utils
    #################################################################

    def display_priorities(self, *, save_dir=None, show=True, feature_names=None):
        """Plot the configured priorities (numerical + categorical).

        Delegates to ``explainit.utils.priority_plots.plot_priorities`` so the
        plotting logic lives in a single reusable place.
        """
        sample = getattr(self.sample_state, "sample", None)
        return _plot_priorities(
            self.priorities_state.priorities,
            sample=sample,
            feature_names=feature_names,
            save_dir=save_dir,
            show=show,
        )
