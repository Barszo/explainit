from explainit.logging_config import logger
from explainit.utils.priority_plots import plot_priorities as _plot_priorities
# logger.info("This is an info message")
# logger.debug("This is a debug message with details")
# logger.warning("This is a warning message")
# logger.error("This is an error message")

import numpy as np
import random
from scipy.optimize import minimize
from math import factorial
import itertools
import copy
from scipy.optimize import linprog
import warnings
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any, Optional, Sequence

@dataclass
class SampleState:

    # original sample that you want to modify to achieve target
    sample : list = None
    # target exemplar from dataset with prediction closest to target
    target_exemplar : list = None
    # Shapley values dictionary calculated between sample and target_exemplar
    shapley_values : Dict[Any, float] = None
    # This dictionary excludes categories that do not change the prediction
    # (shapley value is 0) it is done to limit number of combinations during search
    limited_priorities : Dict[str, Any] = None
    # All combinations of categorical features to consider during search
    all_combinations : Dict[Any, list] = None

@dataclass
class PrioritiesState:
    # TODO: at the end, make sure that the attributes are calculated once only when invoked, and then only when priorities are changed
    priorities: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def numerical_priorities(self) -> Dict[str, Any]:
        return self.priorities.get('numerical', {})
    
    @property
    def categorical_priorities(self) -> Dict[str, Any]:
        return self.priorities.get('categorical', {})

    @property
    def bounds(self) -> Dict[str, Any]:
        bounds = {}
        for idx, val in self.numerical_priorities.items():
            if isinstance(val, dict):
                min_val = val.get('min', None)
                max_val = val.get('max', None)
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
                if val.get('function') is None:
                    non_actionable.append(idx)
            elif val is None:
                # Old format fallback
                non_actionable.append(idx)

        return non_actionable


class MINLSearchExplainer:
    def __init__(self, 
                model_pred, 
                priorities, 
                sample, 
                target, 
                dataset, 
                target_exemplar_epsilon=0.01, 
                epsilon=0.01,
                workflow_logger=None,
                feature_names: Optional[Sequence[str]] = None):
        self.sample_state = SampleState(
            sample = sample
        )
        self.priorities_state = PrioritiesState(
            priorities = priorities
        )
        # TODO: model_pred needs to be prepared before to be used here. It should be changed so that it would be more universal
        self.model_pred = model_pred
        self.target_exemplar_epsilon = target_exemplar_epsilon # Finds element in dataset which prediction is closest to target
                                                            # There is a section in the code below
        self.target = target
        self.dataset = dataset
        self.epsilon = epsilon
        self.workflow_logger = workflow_logger
        self.feature_names = list(feature_names) if feature_names is not None else None
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
                100.0
                if abs(dataset_span) < 1e-12
                else (allowed_span / dataset_span) * 100.0
            )
            if allowed_intervals:
                feature_allowed_mask = np.array([
                    self._in_allowed_intervals(float(v), allowed_intervals)
                    for v in self.dataset[:, idx]
                ], dtype=bool)
            else:
                feature_allowed_mask = np.logical_and(
                    self.dataset[:, idx] >= allowed_min,
                    self.dataset[:, idx] <= allowed_max,
                )
            feature_allowed_pct = 100.0 * float(np.sum(feature_allowed_mask)) / float(n_rows) if n_rows else 0.0
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
        global_allowed_pct = 100.0 * float(np.sum(global_allowed_mask)) / float(n_rows) if n_rows else 0.0
        self._workflow_log("")
        self._workflow_log(
            "Global allowed samples (all numerical-feature bounds simultaneously): %.2f%% (%d/%d)",
            global_allowed_pct,
            int(np.sum(global_allowed_mask)),
            n_rows,
        )

    @staticmethod
    def _in_allowed_intervals(value: float, intervals: list, tol: float = 1e-12) -> bool:
        if not intervals:
            return False
        for lo, hi in intervals:
            if (value >= lo - tol) and (value <= hi + tol):
                return True
        return False

    @staticmethod
    def _allowed_interval_constraint_value(value: float, intervals: list) -> float:
        if not intervals:
            return -1.0
        return max(min(value - lo, hi - value) for lo, hi in intervals)

    @staticmethod
    def _project_to_allowed_intervals(value: float, intervals: list) -> float:
        if not intervals:
            return value
        if MINLSearchExplainer._in_allowed_intervals(value, intervals):
            return value
        candidates = []
        for lo, hi in intervals:
            candidates.append(lo)
            candidates.append(hi)
        return float(min(candidates, key=lambda c: abs(c - value)))

    @staticmethod
    def _extract_positive_intervals(x_grid: np.ndarray, y_grid: np.ndarray, positive_eps: float = 1e-12) -> list:
        mask = np.asarray(y_grid, dtype=float) > float(positive_eps)
        intervals = []
        i = 0
        n = len(x_grid)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            end = i
            intervals.append((float(x_grid[start]), float(x_grid[end])))
            i += 1
        return intervals

    def _derive_bounds_and_intervals_from_priorities(self, grid_size: int = 2000):
        logger.info("--- STAGE 0/6: Derive numerical bounds from priority functions ---")
        logger.info("For each numerical feature, infer feasible ranges from where "
                    "priority function > 0 over dataset support; this overrides user "
                    "min/max and excludes internal zero-priority gaps.")

        for idx, cfg in self.priorities_state.numerical_priorities.items():
            if not isinstance(cfg, dict):
                continue
            fn = cfg.get("function")
            dmin = float(np.min(self.dataset[:, idx]))
            dmax = float(np.max(self.dataset[:, idx]))

            # Non-actionable features remain fixed at their configured value.
            if fn is None:
                fixed_val = float(cfg.get("min", self.sample_state.sample[idx]))
                cfg["min"] = fixed_val
                cfg["max"] = fixed_val
                cfg["allowed_intervals"] = [(fixed_val, fixed_val)]
                continue

            if abs(dmax - dmin) < 1e-12:
                cfg["min"] = dmin
                cfg["max"] = dmax
                cfg["allowed_intervals"] = [(dmin, dmax)]
                continue

            x_grid = np.linspace(dmin, dmax, int(grid_size))
            y_values = []
            for x in x_grid:
                try:
                    y = float(fn(float(x)))
                except Exception:
                    y = 0.0
                if not np.isfinite(y):
                    y = 0.0
                y_values.append(y)
            y_grid = np.array(y_values, dtype=float)
            y_grid[~np.isfinite(y_grid)] = 0.0

            intervals = self._extract_positive_intervals(x_grid, y_grid, positive_eps=1e-12)
            if not intervals:
                raise ValueError(
                    f"Feature {idx} has no positive-priority region on dataset range "
                    f"[{dmin}, {dmax}]. Cannot derive feasible bounds."
                )

            derived_min = float(intervals[0][0])
            derived_max = float(intervals[-1][1])
            cfg["min"] = max(dmin, derived_min)
            cfg["max"] = min(dmax, derived_max)
            cfg["allowed_intervals"] = intervals

            logger.info(
                "[Stage 0] Feature %s (idx=%d): derived bounds=[%.6f, %.6f] "
                "| dataset=[%.6f, %.6f] | allowed_intervals=%s",
                self._feature_label(idx), idx,
                float(cfg["min"]), float(cfg["max"]),
                dmin, dmax,
                intervals,
            )
    
    ################################################################
    # Finds element in dataset which prediction is closest to target
    ################################################################

    ########################################################
    # Filter dataset based on bounds and categorical mapping

    def get_rows_in_priorities(self):
        """
        Filter the dataset to include only samples that satisfy the priority constraints.
        
        This method performs a two-stage filtering process to reduce the dataset to samples 
        that are actionable and within the specified bounds according to the priorities:
        
        1. **Categorical Feature Filtering**: Removes samples with categorical combinations 
           marked as None (non-actionable) in the priorities dictionary
        2. **Numerical Feature Filtering**: Removes samples where numerical features fall 
           outside the min/max bounds specified in the priorities
        
        The filtering is essential for the counterfactual search process as it ensures 
        we only consider samples that:
        - Respect the user's actionability constraints (categorical features)
        - Fall within the user's acceptable ranges (numerical features)
        - Can potentially be reached through valid feature modifications
        
        ## Algorithm Details
        
        **Categorical Filtering**:
        - Identifies categorical combinations with None values in priorities
        - For each unwanted combination, removes all dataset rows matching that exact combination
        - Uses vectorized operations for efficient row removal
        
        **Numerical Filtering**:
        - Applies min/max bounds from the priorities_state.bounds property
        - Filters rows where feature values fall outside [min_val, max_val] ranges
        - Handles cases where only min or max bounds are specified
        
        ## Error Handling
        
        Raises exceptions if filtering results in an empty dataset, which would indicate:
        - Overly restrictive constraints
        - No feasible solutions exist within the specified priorities
        - Potential issues with the priority configuration
        
        Returns:
            numpy.ndarray: Filtered dataset containing only samples that satisfy all 
                         priority constraints. Shape is (n_filtered_samples, n_features)
                         where n_filtered_samples <= original dataset size.
                         
        Raises:
            Exception: If filtering results in an empty dataset at any stage, indicating
                      no samples satisfy the specified priority constraints.
                      
        Example:
            >>> explainer = MINLSearchExplainer(...)
            >>> filtered_data = explainer.get_rows_in_priorities()
            >>> print(f"Dataset reduced from {explainer.dataset.shape[0]} to {filtered_data.shape[0]} samples")
            
        Note:
            This method is automatically called by find_closest_elem() as part of the 
            counterfactual search pipeline. The filtered dataset is used to identify 
            the target exemplar that achieves the target prediction while respecting
            all priority constraints.
        """

        def remv_cat(np_arr, idx_tup,  vals):
            mask = ~(np.all(data_np[:, idx_tup] == vals, axis=1))
            return np_arr[mask]

        logger.info("[Stage 1.1] Filtering dataset (%d rows) using priority "
                    "constraints (categorical None entries + numerical bounds)...",
                    self.dataset.shape[0])
        data_np = self.dataset.copy()

        # Categorical features
        # checking which categorical features are None in priorities
        unwanted_cat_groups = [
            (group, cat_vals)
            for group, mapping in self.priorities_state.categorical_priorities.items()
            for cat_vals, v in mapping.items()
            if v is None
        ]

        # removing samples (rows) that categorical features are None in priorities
        if unwanted_cat_groups:
            logger.info("[Stage 1.1a] Dropping rows matching disallowed categorical "
                        "combos: %d combo(s)", len(unwanted_cat_groups))
        for group, cat_vals in unwanted_cat_groups:
            data_np = remv_cat(data_np, group, cat_vals)
            if data_np.size == 0:
                raise Exception("There are no elements fulfilling the requirements")
        logger.info("[Stage 1.1a] After categorical filtering: %d rows remain.",
                    data_np.shape[0])

        # Numerical features
        # applying min/max bounds from priorities
        # Skip non-actionable features (they don't participate in filtering)
        num_bounds = self.priorities_state.bounds
        non_actionable = self.priorities_state.non_actionable_indices
        for idx, (min_val, max_val) in num_bounds.items():
            # Skip non-actionable features
            if idx in non_actionable:
                continue
            
            if min_val is not None:
                data_np = data_np[data_np[:, idx] >= min_val]
            if max_val is not None:
                data_np = data_np[data_np[:, idx] <= max_val]
            if data_np.size == 0:
                raise Exception("There are no elements fulfilling the requirements")
            cfg = self.priorities_state.numerical_priorities.get(idx, {})
            allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None
            if allowed_intervals:
                in_interval_mask = np.array([
                    self._in_allowed_intervals(float(v), allowed_intervals)
                    for v in data_np[:, idx]
                ], dtype=bool)
                data_np = data_np[in_interval_mask]
                if data_np.size == 0:
                    raise Exception("There are no elements fulfilling the requirements")
        logger.info("[Stage 1.1b] After numerical bound filtering: %d rows remain.",
                    data_np.shape[0])

        return data_np

    ################################################################
    # Finds element in dataset (1)
    ################################################################

    def find_closest_elem(self) -> list:
        """
        Finds the elements in the dataset that are closest to the desired target value +- epsilon.
        Returns the actual samples of all such elements.
        """
        logger.info("[Stage 1.2] Searching the priority-filtered dataset for the row "
                    "whose model prediction best matches target=%.4f...",
                    float(self.target))
        filtered_data = self.get_rows_in_priorities()
        pred = self.model_pred(filtered_data)

        min_dist = np.abs(pred - self.target).min()
        logger.info("[Stage 1.2] Closest filtered-row prediction is %.4f away from target.",
                    float(min_dist))
        if min_dist > self.target_exemplar_epsilon:
            raise ValueError(f"No elements found within the specified epsilon of {self.target_exemplar_epsilon}. Closest distance is {min_dist}.")

        idx_all = np.where(np.abs(pred - self.target) == min_dist)[0]
        assert len(idx_all) == 1, "Expected exactly one target exemplar index, but found multiple."
        self.sample_state.target_exemplar = filtered_data[idx_all[0]]
        logger.info("[Stage 1.2] Target exemplar locked in (filtered idx=%d, "
                    "prediction=%.4f).",
                    int(idx_all[0]), float(pred[idx_all[0]]))
    
    ##################################
    # Calculating Shapley values (3)
    ##################################

    def calc_shapley(self, r: list, use_approximation: bool = False, num_samples: int = 200) -> list:
        """
        Calculate Shapley values for feature importance in counterfactual explanations.
        
        This implementation treats categorical feature groups as single units for Shapley 
        calculation, avoiding redundant calculations for one-hot encoded features.
        """
        cat_groups = [list(elem) for elem in self.priorities_state.categorical_priorities.keys()]

        def create_feature_mapping():
            """Create mapping from original features to Shapley calculation units"""
            feature_map = {}
            shapley_idx = 0
            
            # Map numerical features (1:1 mapping)
            for num_idx in self.priorities_state.numerical_priorities.keys():
                feature_map[num_idx] = shapley_idx
                shapley_idx += 1
            
            # Map categorical groups (many:1 mapping)
            cat_to_shapley = {}
            for cat_group in cat_groups:
                for cat_idx in cat_group:
                    feature_map[cat_idx] = shapley_idx
                cat_to_shapley[tuple(cat_group)] = shapley_idx
                shapley_idx += 1
            
            return feature_map, cat_to_shapley, shapley_idx

        def create_consolidated_vectors(original_r, original_x, feature_map, total_shapley_features):
            """Convert original feature vectors to consolidated representation"""
            consolidated_r = [0] * total_shapley_features
            consolidated_x = [0] * total_shapley_features
            
            # Handle numerical features
            for num_idx in self.priorities_state.numerical_priorities.keys():
                shapley_idx = feature_map[num_idx]
                consolidated_r[shapley_idx] = original_r[num_idx]
                consolidated_x[shapley_idx] = original_x[num_idx]
            
            # Handle categorical groups - represent as tuples
            for cat_group in cat_groups:
                shapley_idx = feature_map[cat_group[0]]  # All indices in group map to same shapley_idx
                r_values = tuple(original_r[i] for i in cat_group)
                x_values = tuple(original_x[i] for i in cat_group)
                consolidated_r[shapley_idx] = r_values
                consolidated_x[shapley_idx] = x_values
            
            return consolidated_r, consolidated_x

        def expand_to_original_format(consolidated_vector, feature_map, original_length):
            """Expand consolidated vector back to original feature space"""
            expanded = [0] * original_length
            
            # Handle numerical features
            for num_idx in self.priorities_state.numerical_priorities.keys():
                shapley_idx = feature_map[num_idx]
                expanded[num_idx] = consolidated_vector[shapley_idx]
            
            # Handle categorical groups
            for cat_group in cat_groups:
                shapley_idx = feature_map[cat_group[0]]
                cat_values = consolidated_vector[shapley_idx]
                if isinstance(cat_values, tuple):
                    for i, cat_idx in enumerate(cat_group):
                        expanded[cat_idx] = cat_values[i]
                else:
                    # Fallback: use same value for all in group
                    for cat_idx in cat_group:
                        expanded[cat_idx] = cat_values
            
            return expanded

        def z_of_S_batch_consolidated(S_list, consolidated_r, consolidated_x, feature_map, original_length):
            """Create batch of hybrid vectors using consolidated representation"""
            z_batch = []
            
            for S in S_list:
                # Start with reference vector
                z_consolidated = consolidated_r.copy()
                
                # Replace features in S with values from x
                for i in S:
                    z_consolidated[i] = consolidated_x[i]
                
                # Expand back to original format for model prediction
                z_expanded = expand_to_original_format(z_consolidated, feature_map, original_length)
                z_batch.append(z_expanded)
            
            return np.array(z_batch)

        def shapley_value_vectorized_consolidated(i, consolidated_x, consolidated_r, f, n, feature_map, original_length):
            """Vectorized Shapley calculation for consolidated features"""
            others = [j for j in range(n) if j != i]
            
            all_subsets = []
            all_subsets_with_i = []
            all_weights = []
            
            total_subsets = 2 ** len(others)
            logger.info("[Stage 3] Exact Shapley for unit %d/%d: enumerating %d subsets "
                        "(O(2^(n-1)) work).",
                        i + 1, n, total_subsets)
            
            for k in range(len(others) + 1):
                for S in itertools.combinations(others, k):
                    S = set(S)
                    weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    
                    all_subsets.append(S)
                    all_subsets_with_i.append(S | {i})
                    all_weights.append(weight)
            
            # Create batches for prediction
            z_without_batch = z_of_S_batch_consolidated(all_subsets, consolidated_r, consolidated_x, feature_map, original_length)
            z_with_batch = z_of_S_batch_consolidated(all_subsets_with_i, consolidated_r, consolidated_x, feature_map, original_length)
            
            # Batch predictions
            pred_without_batch = f(z_without_batch)
            pred_with_batch = f(z_with_batch)
            
            # Ensure proper shape
            if pred_without_batch.ndim > 1:
                pred_without_batch = pred_without_batch.flatten()
            if pred_with_batch.ndim > 1:
                pred_with_batch = pred_with_batch.flatten()
            
            # Calculate weighted differences
            diff_batch = pred_with_batch - pred_without_batch
            weighted_contributions = np.array(all_weights) * diff_batch
            
            total = np.sum(weighted_contributions)
            return total

        def shapley_value_approximate_consolidated(i, consolidated_x, consolidated_r, f, n, feature_map, original_length, num_samples=200):
            """Approximate Shapley calculation for consolidated features"""
            others = [j for j in range(n) if j != i]
            
            logger.info("[Stage 3] Approximate Shapley for unit %d/%d: averaging %d "
                        "random subsets (Monte-Carlo estimator).",
                        i + 1, n, num_samples)
            
            all_subsets = []
            all_subsets_with_i = []
            
            for _ in range(num_samples):
                subset_size = random.randint(0, len(others))
                if subset_size == 0:
                    S = set()
                elif subset_size == len(others):
                    S = set(others)
                else:
                    S = set(random.sample(others, subset_size))
                
                all_subsets.append(S)
                all_subsets_with_i.append(S | {i})
            
            # Create batches for prediction
            z_without_batch = z_of_S_batch_consolidated(all_subsets, consolidated_r, consolidated_x, feature_map, original_length)
            z_with_batch = z_of_S_batch_consolidated(all_subsets_with_i, consolidated_r, consolidated_x, feature_map, original_length)
            
            # Batch predictions
            pred_without_batch = f(z_without_batch)
            pred_with_batch = f(z_with_batch)
            
            # Ensure proper shape
            if pred_without_batch.ndim > 1:
                pred_without_batch = pred_without_batch.flatten()
            if pred_with_batch.ndim > 1:
                pred_with_batch = pred_with_batch.flatten()
            
            # Calculate average marginal contribution
            diff_batch = pred_with_batch - pred_without_batch
            total = np.mean(diff_batch)
            
            return total

        # Main calculation logic
        original_r = r
        original_x = self.sample_state.target_exemplar
        f = self.model_pred
        original_length = len(original_r)

        # Create feature mapping and consolidated vectors
        feature_map, cat_to_shapley, total_shapley_features = create_feature_mapping()
        consolidated_r, consolidated_x = create_consolidated_vectors(original_r, original_x, feature_map, total_shapley_features)

        logger.info("[Stage 3] Consolidating features for Shapley: %d original cols -> "
                    "%d units (numerical kept 1:1, each one-hot group merged).",
                    original_length, total_shapley_features)
        logger.info("[Stage 3] Numerical features tracked: %d | Categorical groups: %d",
                    len(self.priorities_state.numerical_priorities), len(cat_groups))

        # Calculate Shapley values for consolidated features
        if use_approximation:
            phi = [shapley_value_approximate_consolidated(i, consolidated_x, consolidated_r, f, total_shapley_features, feature_map, original_length, num_samples) 
                   for i in range(total_shapley_features)]
        else:
            phi = [shapley_value_vectorized_consolidated(i, consolidated_x, consolidated_r, f, total_shapley_features, feature_map, original_length) 
                   for i in range(total_shapley_features)]

        # Create dictionary mapping feature indices/groups to their Shapley values
        shapley_dict = {
            'numerical': {},
            'categorical': {}
        }
        shapley_idx = 0
        
        # Map numerical features (single indices as keys)
        for num_idx in self.priorities_state.numerical_priorities.keys():
            shapley_dict['numerical'][num_idx] = phi[shapley_idx]
            shapley_idx += 1
        
        # Map categorical groups (tuples of indices as keys)
        for cat_group in cat_groups:
            shapley_dict['categorical'][tuple(cat_group)] = phi[shapley_idx]
            shapley_idx += 1
        
        # Store in SampleState
        self.sample_state.shapley_values = shapley_dict
        
        return np.array(phi, dtype=float)


    #################################################################
    # Confirm existence of solution for each categorical combination (4)
    #################################################################

    # Helper function to find_required_bounds_adjustments
    def _validate_bounds_adjustments(self, required_shifts, cat_combinations, shap_dict, 
                                   priorities_for_search, basic_prediction, indices_to_modify, 
                                   coeff_to_linear_search, target_for_combo):
        """
        Validate that the proposed bounds adjustments actually enable feasible solutions.
        
        Tests the adjusted bounds with the actual constraint function to ensure the
        linear approximation-based suggestions work in practice.
        """
        # Create adjusted bounds
        adjusted_bounds = []
        for i, feature_idx in enumerate(indices_to_modify):
            if feature_idx in required_shifts:
                new_bounds = required_shifts[feature_idx]['suggested_bounds']
                logger.info(f"Adjusting bounds for feature {feature_idx}: {new_bounds}")
            else:
                # Keep original bounds for features that don't need adjustment
                original_priorities = priorities_for_search['numerical'][feature_idx]
                new_bounds = (original_priorities['min'], original_priorities['max'])
            adjusted_bounds.append(new_bounds)
        
        successful_validations = 0
        total_attempts = 0
        
        # Test all categorical combinations with adjusted bounds
        for combination_id, combo in cat_combinations.items():
            total_attempts += 1
            
            # Get the pre-calculated target for this combination
            if combination_id not in target_for_combo:
                continue
                
            target = target_for_combo[combination_id]
            logger.info(f"Validating combination {combination_id} with target {target}")
            
            # Try to solve with adjusted bounds
            linear_solution = MINLSearchExplainer.solve_linear_constraint_lp(
                coeff_to_linear_search, target, adjusted_bounds, 
                method='auto', tolerance=self.epsilon
            )
            
            if linear_solution['success']:
                logger.info(f"LP solution found for combination {combination_id}")
                
                # Validate with actual model
                dummy_x = self.sample.copy()
                
                # Set categorical features
                for idx, val in combo.items():
                    dummy_x[idx] = val
                
                # Set numerical features from LP solution
                for i, feature_idx in enumerate(indices_to_modify):
                    dummy_x[feature_idx] = linear_solution['solution'][i]
                    logger.info(f"Set feature {feature_idx} to {linear_solution['solution'][i]} (bounds: {adjusted_bounds[i]})")
                
                # Check constraint with actual model
                try:
                    constraint_value = self.constraint_function(
                        dummy_x, shap_dict, self.sample, self.target_exemplar, 
                        priorities_for_search, basic_prediction
                    )
                    
                    error = abs(constraint_value - self.target)
                    logger.info(f"Validation for combination {combination_id}: constraint={constraint_value:.3f}, target={self.target:.3f}, error={error:.3f}")
                    
                    if error <= self.epsilon:
                        successful_validations += 1
                        logger.info(f"Validation SUCCESSFUL for combination {combination_id}")
                    else:
                        # Even if linear approximation fails, if we're reasonably close, consider it partially successful
                        if error <= 3 * self.epsilon:  # Allow 3x tolerance for validation
                            successful_validations += 0.5  # Partial success
                            logger.info(f"Validation PARTIALLY SUCCESSFUL for combination {combination_id}")
                        else:
                            logger.info(f"Validation FAILED for combination {combination_id}")
                        
                except Exception as e:
                    logger.warning(f"Validation error for combination {combination_id}: {e}")
            else:
                logger.info(f"LP solution failed for combination {combination_id}: {linear_solution.get('message', 'Unknown error')}")
        
        # Calculate confidence based on validation success rate
        if total_attempts > 0:
            confidence = successful_validations / total_attempts
            feasible = confidence > 0  # At least some validation succeeded
        else:
            confidence = 0.0
            feasible = len(required_shifts) > 0  # Assume feasible if shifts were identified
        
        logger.info(f"Validation summary: {successful_validations}/{total_attempts} successful, confidence={confidence:.2f}")
        return feasible, confidence
    
    # Helper function to find_required_bounds_adjustments
    def _solve_unconstrained_linear_system(self, coefficients, target):
        """
        Solve the linear system without bounds constraints to find optimal feature values.
        
        For the system: c₁x₁ + c₂x₂ + ... + cₙxₙ = target
        We need to find one solution. We use least-norm solution when underdetermined.
        """
        coefficients = np.array(coefficients)
        
        # Handle case where all coefficients are zero
        if np.allclose(coefficients, 0):
            return None
            
        # For underdetermined system, find minimum norm solution
        try:
            # Use pseudoinverse to find least-squares solution
            A = coefficients.reshape(1, -1)  # Shape: (1, n)
            b = np.array([target])  # Shape: (1,)
            
            solution = np.linalg.pinv(A) @ b
            return solution
            
        except np.linalg.LinAlgError:
            # If system is inconsistent, find least-squares approximation
            try:
                solution = np.linalg.lstsq(A, b, rcond=None)[0]
                return solution
            except:
                return None
    
    # Helper function to find_required_bounds_adjustments
    def _find_single_feature_solutions(self, combination_id, target, indices_to_modify, 
                                     coefficients, original_bounds):
        """
        Find solutions that change only one feature at a time.
        
        This method is often more practical than multi-feature solutions and matches
        how users often think about the problem (e.g., "just increase income").
        """
        single_feature_solutions = {}
        
        for i, feature_idx in enumerate(indices_to_modify):
            coeff = coefficients[i]
            current_bounds = original_bounds[i]
            min_bound, max_bound = current_bounds
            
            if abs(coeff) < 1e-10:  # Skip features with near-zero coefficients
                continue
                
            # Calculate required feature value to achieve target by changing only this feature
            required_value = target / coeff
            
            logger.info(f"Feature {feature_idx}: coeff={coeff:.6f}, required_value={required_value:.6f}, bounds=({min_bound:.6f}, {max_bound:.6f})")
            
            requires_shift = False
            shift_magnitude = 0
            new_min, new_max = min_bound, max_bound
            justification = ""
            
            if required_value < min_bound:
                shift_magnitude = min_bound - required_value
                new_min = required_value - 0.1 * abs(required_value + 1e-6)
                justification = f"Single-feature solution needs value {required_value:.3f}, below current min {min_bound:.3f}"
                requires_shift = True
                
            elif required_value > max_bound:
                shift_magnitude = required_value - max_bound
                new_max = required_value + 0.1 * abs(required_value + 1e-6)
                justification = f"Single-feature solution needs value {required_value:.3f}, above current max {max_bound:.3f}"
                requires_shift = True
            else:
                justification = f"Single-feature solution feasible with value {required_value:.3f}"
            
            single_feature_solutions[feature_idx] = {
                'required_value': required_value,
                'current_bounds': current_bounds,
                'required_bounds': (new_min, new_max),
                'requires_bounds_shift': requires_shift,
                'shift_magnitude': shift_magnitude,
                'justification': justification,
                'coefficient': coeff
            }
        
        return single_feature_solutions
    

    # If no initial feasible solutions were found, analyze why and suggest bounds adjustments (4.6)
    def find_required_bounds_adjustments(self, cat_combinations, shap_dict, priorities_for_search, 
                                       basic_prediction, target_for_combo, shap_coefficients):
        """
        Intelligently determine how bounds should be adjusted to enable feasible solutions.
        
        This function analyzes why no feasible solutions were found and calculates the minimum
        bounds adjustments needed to make the problem feasible. It uses both the linear approximation
        and direct search methods to identify which features need expanded bounds.
        
        Algorithm:
        1. Try single-feature solutions to identify which features can solve the constraint alone
        2. Calculate minimum bounds shifts needed for feasibility
        3. Validate suggestions using actual model constraint function
        4. Provide actionable recommendations for bounds adjustment
        
        Args:
            cat_combinations: Dictionary of categorical feature combinations
            shap_dict: Shapley values organized by feature type
            priorities_for_search: Current priorities with bounds
            basic_prediction: Model prediction for original sample
            target_for_combo: Target values for each categorical combination
            shap_coefficients: Linear coefficients from Shapley analysis
            
        Returns:
            dict: {
                'feasible_with_shifts': bool,
                'required_shifts': {feature_idx: {'original_bounds', 'suggested_bounds', 'justification'}},
                'affected_combinations': list,
                'confidence': float
            }
        """
        logger.info("Analyzing bounds constraints to find required adjustments...")
        
        indices_to_modify = list(shap_coefficients.keys())
        coeff_to_linear_search = [shap_coefficients[idx]['coeff'] for idx in indices_to_modify]
        original_bounds = [shap_coefficients[idx]['min_max'] for idx in indices_to_modify]
        
        required_shifts = {}
        feasible_combinations = []
        
        # Analyze each categorical combination
        for combination_id, temp_target in target_for_combo.items():
            logger.info(f"Analyzing infeasibility for combination {combination_id}")
            logger.info(f"Target to achieve: {temp_target}")
            
            # Method 1: Try single-feature solutions (often most practical)
            single_feature_solutions = self._find_single_feature_solutions(
                combination_id, temp_target, indices_to_modify, 
                coeff_to_linear_search, original_bounds
            )
            
            # Method 2: Try unconstrained multi-feature solution
            unconstrained_solution = self._solve_unconstrained_linear_system(
                coeff_to_linear_search, temp_target
            )
            
            # Combine results from both methods
            all_candidate_shifts = {}
            
            # Process single-feature solutions
            for feature_idx, info in single_feature_solutions.items():
                if info['requires_bounds_shift']:
                    all_candidate_shifts[feature_idx] = {
                        'original_bounds': info['current_bounds'],
                        'suggested_bounds': info['required_bounds'],
                        'justification': info['justification'],
                        'shift_magnitude': info['shift_magnitude'],
                        'method': 'single_feature'
                    }
            
            # Process unconstrained solution
            if unconstrained_solution is not None:
                for i, feature_idx in enumerate(indices_to_modify):
                    optimal_value = unconstrained_solution[i]
                    current_bounds = original_bounds[i]
                    min_bound, max_bound = current_bounds
                    
                    shift_needed = 0
                    new_min, new_max = min_bound, max_bound
                    
                    if optimal_value < min_bound:
                        shift_needed = min_bound - optimal_value
                        new_min = optimal_value - 0.05 * abs(optimal_value + 1e-6)
                        justification = f"Multi-feature solution needs decrease by {shift_needed:.3f}"
                        
                    elif optimal_value > max_bound:
                        shift_needed = optimal_value - max_bound
                        new_max = optimal_value + 0.05 * abs(optimal_value + 1e-6)
                        justification = f"Multi-feature solution needs increase by {shift_needed:.3f}"
                    
                    if shift_needed > 0:
                        # Compare with single-feature solution if it exists
                        if feature_idx in all_candidate_shifts:
                            existing_shift = all_candidate_shifts[feature_idx]['shift_magnitude']
                            if shift_needed < existing_shift:  # Prefer smaller shifts
                                all_candidate_shifts[feature_idx] = {
                                    'original_bounds': current_bounds,
                                    'suggested_bounds': (new_min, new_max),
                                    'justification': justification,
                                    'shift_magnitude': shift_needed,
                                    'method': 'multi_feature'
                                }
                        else:
                            all_candidate_shifts[feature_idx] = {
                                'original_bounds': current_bounds,
                                'suggested_bounds': (new_min, new_max),
                                'justification': justification,
                                'shift_magnitude': shift_needed,
                                'method': 'multi_feature'
                            }
            
            # Update global required_shifts with best options
            for feature_idx, shift_info in all_candidate_shifts.items():
                if feature_idx not in required_shifts:
                    required_shifts[feature_idx] = shift_info.copy()
                    required_shifts[feature_idx]['combinations_affected'] = [combination_id]
                else:
                    # Keep the shift that requires smaller adjustment
                    existing_shift = required_shifts[feature_idx]['shift_magnitude']
                    if shift_info['shift_magnitude'] < existing_shift:
                        required_shifts[feature_idx] = shift_info.copy()
                        required_shifts[feature_idx]['combinations_affected'] = [combination_id]
                    required_shifts[feature_idx]['combinations_affected'].append(combination_id)
        
        # Step 3: Validate suggestions using actual model if any shifts were identified
        confidence = 0.0
        feasible_with_shifts = False
        
        if required_shifts:
            feasible_with_shifts, confidence = self._validate_bounds_adjustments(
                required_shifts, cat_combinations, shap_dict, priorities_for_search, 
                basic_prediction, indices_to_modify, coeff_to_linear_search, target_for_combo
            )
            
            # Get list of combinations that become feasible
            all_affected_combinations = set()
            for shift_info in required_shifts.values():
                all_affected_combinations.update(shift_info['combinations_affected'])
            feasible_combinations = list(all_affected_combinations)
        
        return {
            'feasible_with_shifts': feasible_with_shifts,
            'required_shifts': required_shifts,
            'affected_combinations': feasible_combinations,
            'confidence': confidence
        }
    
    # Helper function to solve_linear_constraint_lp. Fallback corner analysis method for LP solving
    @staticmethod
    def _fallback_corner_analysis(coefficients, target, bounds, tolerance):
        """Fallback method using corner point analysis"""
        corners = []
        for bound in bounds:
            corners.append([bound[0], bound[1]])
        
        import itertools

        corner_points = list(itertools.product(*corners))
        
        best_error = float('inf')
        best_solution = None
        
        for corner in corner_points:
            corner = np.array(corner)
            value = np.dot(coefficients, corner)
            error = abs(value - target)
            
            if error < best_error:
                best_error = error
                best_solution = corner
        
        if best_error < tolerance:
            return {
                'success': True,
                'solution': best_solution,
                'message': f'Corner solution found with error {best_error:.2e}',
                'method_used': 'corner_analysis',
                'constraint_value': np.dot(coefficients, best_solution),
                'error': best_error
            }
        else:
            return {
                'success': False,
                'solution': None,
                'message': f'No feasible solution found. Best error: {best_error:.2e}',
                'method_used': 'corner_analysis'
            }


    # function for finding initial solution for counterfactual search (4.5)
    @staticmethod
    def solve_linear_constraint_lp(coefficients, target, bounds, method='auto', tolerance=1e-8):
        """
        Production-ready linear programming solver for constraint satisfaction
        
        Args:
            coefficients: array-like, coefficients [a₁, a₂, ..., aₙ]
            target: float, target value for a₁x₁ + a₂x₂ + ... + aₙxₙ = target
            bounds: list of tuples, [(min₁, max₁), (min₂, max₂), ...]
            method: str, LP algorithm ('highs', 'interior-point', 'revised simplex')
            tolerance: float, tolerance for constraint satisfaction
        
        Returns:
            dict: {'success': bool, 'solution': array, 'message': str, 'method_used': str}
        """
        coefficients = np.array(coefficients)
        n = len(coefficients)
        
        # Input validation
        if len(bounds) != n:
            return {'success': False, 'solution': None, 
                    'message': 'Bounds length must match coefficients length', 'method_used': None}
        
        # Check if all coefficients are zero
        if np.allclose(coefficients, 0):
            if abs(target) < tolerance:
                # Any point in bounds is a solution
                solution = np.array([(b[0] + b[1])/2 for b in bounds])
                return {'success': True, 'solution': solution, 
                    'message': 'Trivial solution (all coefficients zero)', 'method_used': 'analytical'}
            else:
                return {'success': False, 'solution': None, 
                    'message': 'No solution (all coefficients zero, target non-zero)', 'method_used': None}
        
        # Try different methods if auto is selected
        methods_to_try = ['highs', 'interior-point'] if method == 'auto' else [method]
        
        for current_method in methods_to_try:
            try:
                # Set up LP problem
                # Convert equality to two inequalities
                A_ub = np.array([coefficients, -coefficients])
                b_ub = np.array([target, -target])
                
                # Dummy objective (just find feasible point)
                c = np.zeros(n)
                
                # Suppress warnings for cleaner output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = linprog(
                        c=c,
                        A_ub=A_ub,
                        b_ub=b_ub,
                        bounds=bounds,
                        method=current_method,
                        options={'presolve': True}
                    )
                
                if result.success:
                    # Verify solution
                    constraint_value = np.dot(coefficients, result.x)
                    error = abs(constraint_value - target)
                    
                    if error < tolerance:
                        return {
                            'success': True,
                            'solution': result.x,
                            'message': f'Solution found with error {error:.2e}',
                            'method_used': current_method,
                            'constraint_value': constraint_value,
                            'error': error
                        }
            
            except Exception as e:
                continue
        
        # If no method worked, try corner analysis
        return MINLSearchExplainer._fallback_corner_analysis(coefficients, target, bounds, tolerance)

    # Calculating shap cefficients and new targets for linear search (4.3)
    def extract_for_linear_search(self):
        """
        Transform the nonlinear counterfactual search problem into linear programming subproblems.
        
        This function is the core mathematical engine that converts the complex counterfactual 
        optimization problem into manageable linear programming problems. It uses Shapley values 
        to create linear approximations of how feature changes affect model predictions.
        
        ## Mathematical Foundation
        
        The function approximates the constraint equation:
        ```
        model_prediction(x) ≈ basic_prediction + Σ(unit_phi_i × (x_i - sample_i)) + categorical_contributions
        ```
        
        Where:
        - `unit_phi_i` is the per-unit Shapley contribution for numerical feature i
        - `categorical_contributions` are discrete Shapley values for categorical combinations
        
        ## The Transformation Process
        
        1. **Numerical Features**: Converts Shapley values into linear coefficients
           - For each numerical feature i: `unit_phi_i = shapley_i / (target_exemplar_i - sample_i)`
           - This gives the linear rate of change in prediction per unit change in feature
        
        2. **Categorical Features**: Calculates discrete contributions
           - Each categorical combination contributes a fixed Shapley value
           - Only valid combinations from `cat_combinations` are considered
        
        3. **Target Adjustment**: Creates adjusted targets for linear programming
           - Original target: `self.target`
           - Subtract categorical contributions and baseline effects
           - Result: target value that numerical features must achieve
        
        ## Why This Decomposition Works
        
        By fixing categorical features to specific combinations, the problem becomes:
        ```
        Find numerical features x such that:
        Σ(coefficient_i × x_i) = adjusted_target
        ```
        
        This is a standard linear programming constraint that can be solved efficiently.
        
        ## Example Workflow
        
        Suppose we want model prediction = 0.8, current prediction = 0.3:
        1. Fix categorical features to combination A
        2. Calculate categorical contribution: +0.2
        3. Remaining gap to fill with numerical features: 0.8 - 0.3 - 0.2 = 0.3
        4. Use Shapley coefficients to find numerical values achieving +0.3 prediction change
        
        Args:
            cat_combinations (dict): Dictionary mapping combination IDs to categorical feature 
                                   value dictionaries. Format: {combo_id: {feature_idx: value}}
            shap_dict (dict): Shapley values organized as {'numerical': {idx: shap_val}, 
                             'categorical': {feature_indices: shap_val}}
            priorities_for_search (dict): Modified priorities containing actionable features with
                                        bounds. Format: {'numerical': {idx: {'min': val, 'max': val}}}
            basic_prediction (float): Model's prediction for the original sample
            
        Returns:
            tuple: (new_targets, shap_coeffs)
                - new_targets (dict): Mapping from combination ID to adjusted target value
                  that numerical features must achieve through linear programming
                - shap_coeffs (dict): Linear coefficients and bounds for numerical features.
                  Format: {feature_idx: {'coeff': linear_coefficient, 'min_max': (min, max)}}
                  
        Raises:
            ValueError: If an invalid categorical combination is encountered that doesn't
                       match either the sample or target_exemplar values
                       
        Example:
            >>> cat_combos = {0: {3: 1.0, 4: 0.0}, 1: {3: 0.0, 4: 1.0}}
            >>> shap_dict = {'numerical': {0: 0.1, 1: -0.2}, 'categorical': {(3,4): 0.15}}
            >>> targets, coeffs = explainer.extract_for_linear_search(cat_combos, shap_dict, 
            ...                                                      priorities, 0.5)
            >>> print(targets)  # {0: 0.23, 1: 0.08}  # Different targets per combination
            >>> print(coeffs)   # {0: {'coeff': 2.5, 'min_max': (0, 1)}, ...}
        """

        num_shap = self.sample_state.shapley_values['numerical']
        cat_shap = self.sample_state.shapley_values['categorical']
        priorities_for_search = self.sample_state.limited_priorities
        cat_combinations = self.sample_state.all_combinations
        basic_prediction = self.basic_prediction

        new_targets = {}
        coef_times_original = 0.0
        shap_coeffs = {}

        # calcualting coefficients for numerical features
        for key, value in self.priorities_state.numerical_priorities.items():
            denom = (self.sample_state.target_exemplar[key] - self.sample_state.sample[key])
            # If the feature value does not change between sample and target_exemplar,
            # the unit coefficient is undefined; treat it as 0 to avoid NaNs.
            if abs(float(denom)) < 1e-12:
                temp_unit_phi = 0.0
            else:
                temp_unit_phi = num_shap[key] / denom
            shap_coeffs[key] = temp_unit_phi # save coefficient, even for non-actionable features
            
            # Only include actionable features in coef_times_original
            # Non-actionable features don't change, so they contribute 0 to the linear constraint
            # The constraint function handles non-actionable features separately (x_i = sample_i always)
            if key not in self.priorities_state.non_actionable_indices:
                coef_times_original += temp_unit_phi * self.sample_state.sample[key]

        self.sample_state.shap_coeffs = shap_coeffs

        if len(cat_combinations[0]) != 0:
        #if there are categorical features
            for combo_idx, combo in cat_combinations.items():
                result = 0.0
                if not combo:
                    print("Empty combination, skipping")
                    continue

                # ensure array is large enough for the highest index (max index is inclusive)
                max_idx = max(combo.keys())
                dummy_x = np.zeros(max_idx + 1, dtype=float)

                for idx, val in combo.items():
                    dummy_x[idx] = val

                for key in self.sample_state.limited_priorities['categorical'].keys():
                    feature_indices = key
                    shap_value = cat_shap[feature_indices]
                
                    # Extract current values from x
                    current_values = tuple(float(dummy_x[idx]) for idx in feature_indices)
                    sample_values = tuple(float(self.sample_state.sample[idx]) for idx in feature_indices)
                    target_exemplar_values = tuple(float(self.sample_state.target_exemplar[idx]) for idx in feature_indices)
                
                    if current_values == sample_values:
                        result += 0.0
                    elif current_values == target_exemplar_values:
                        result += shap_value
                    else:
                        raise ValueError("Invalid categorical combination encountered for indices {}: {}".format(feature_indices, current_values))

                new_targets[combo_idx] = self.target - result - basic_prediction + coef_times_original #- unactionable_sum
        else:
            # no categorical features
            new_targets[0] = self.target - basic_prediction + coef_times_original

        return new_targets

    # function to create limited priorities based on SHAP values (4.1)
    def create_limited_priorities(self):
        """
        Create modified priorities dictionary based on SHAP values and sample data.
        
        This method processes the calculated Shapley values to determine which categorical 
        combinations should be considered during the counterfactual search. The logic is:
        
        For categorical features:
        - If SHAP value is 0: Feature doesn't contribute to prediction difference, 
          so keep only the category from the original sample
        - If SHAP value is not 0: Feature contributes to prediction difference, 
          so allow categories from both sample and target_exemplar
        
        This reduces the search space by eliminating categorical combinations that 
        don't contribute to achieving the target prediction.
        
        Results are stored in:
            - sample_state.limited_priorities: Updated priorities with filtered categorical options
            - sample_state.all_combinations: All possible categorical combinations to explore
        """

        # Create a deep copy of the original priorities
        modified_priorities = copy.deepcopy(self.priorities_state.priorities)
        
        # Use existing Shapley values from sample_state
        shap_values_dict = self.sample_state.shapley_values
        
        # Store allowed combinations for each categorical group
        allowed_combinations_by_group = []
        all_categorical_indices = []
        
        # Process categorical features
        for feature_indices, categorical_weights in self.priorities_state.categorical_priorities.items():
            # Extract values for this categorical group from sample and target_exemplar
            sample_values = tuple(float(self.sample_state.sample[idx]) for idx in feature_indices)
            target_exemplar_values = tuple(float(self.sample_state.target_exemplar[idx]) for idx in feature_indices)
            
            # Get SHAP value for this categorical group
            shap_val = shap_values_dict['categorical'].get(feature_indices, 0.0)
            
            # Create new categorical weights dictionary and collect allowed combinations
            new_categorical_weights = {}
            group_combinations = []
            
            if shap_val == 0:
                # Keep only the category from sample
                # This is to exclude situations where this feature is the same for both sample and target_exemplar
                # It would be useless to include both if they are identical
                if sample_values in categorical_weights:
                    new_categorical_weights[sample_values] = categorical_weights[sample_values]
                    group_combinations.append(sample_values)
            else:
                # Keep categories from both sample and target_exemplar
                if sample_values in categorical_weights:
                    new_categorical_weights[sample_values] = categorical_weights[sample_values]
                    group_combinations.append(sample_values)
                if target_exemplar_values in categorical_weights and target_exemplar_values != sample_values:
                    new_categorical_weights[target_exemplar_values] = categorical_weights[target_exemplar_values]
                    group_combinations.append(target_exemplar_values)
            
            # Update the modified priorities
            modified_priorities['categorical'][feature_indices] = new_categorical_weights
            
            # Store for cross-product calculation
            allowed_combinations_by_group.append(group_combinations)
            all_categorical_indices.extend(feature_indices)
        
        # Generate all possible combinations across all categorical groups
        all_combinations = {}
        combination_id = 0
        
        # Use itertools.product to get all combinations across groups
        for combination_tuple in itertools.product(*allowed_combinations_by_group):
            # Create dictionary mapping each categorical index to its value
            combination_dict = {}
            
            # Track position in the flattened combination
            group_index = 0
            
            for feature_indices in self.priorities_state.categorical_priorities.keys():
                group_values = combination_tuple[group_index]
                for i, idx in enumerate(feature_indices):
                    combination_dict[idx] = group_values[i]
                group_index += 1
            
            all_combinations[combination_id] = combination_dict
            combination_id += 1
        
        # Store results in SampleState
        self.sample_state.limited_priorities = modified_priorities
        self.sample_state.all_combinations = all_combinations

    # function to confirm existence of solution for each categorical combination
    def confirm_existence_of_solution_for_combo(self):

        # 1. Prepare variables for next steps, including categorical combinations
        # limiting the search space based on SHAP values
        # exclude categories that do not contribute to reaching the target (according to Shapley values)
        # creates list of categorical combinations to explore
        # creates self.sample_state.limited_priorities and self.sample_state.all_combinations
        # now, there are at most two categories in each categorical group (one from sample, one from target_exemplar)
        logger.info("[Stage 4.1] Pruning categorical combinations using Shapley values "
                    "(zero-impact groups collapse to the sample's own category).")
        self.create_limited_priorities()
        logger.info("[Stage 4.1] %d categorical combination(s) survive after pruning.",
                    len(self.sample_state.all_combinations))

        # self.shap_dict = shap_dict
        # 2. Extract targets for linear search for each categorical combination
        logger.info("[Stage 4.2] Caching baseline prediction for the original sample "
                    "(used as anchor in the linear approximation).")
        self.basic_prediction=self.model_pred([self.sample_state.sample])[0]
        logger.info("[Stage 4.2] basic_prediction=%.4f (target=%.4f)",
                    float(self.basic_prediction), float(self.target))

        # 3. Prepare targets and coefficients for linear search
        logger.info("[Stage 4.3] Building per-combo LP targets and per-feature linear "
                    "coefficients from Shapley values.")
        target_for_combo = self.extract_for_linear_search()
        logger.info("[Stage 4.3] LP targets per categorical combo: %s", target_for_combo)
        logger.debug("[Stage 4.3] Per-feature shap_coeffs (unit phi): %s",
                     self.sample_state.shap_coeffs)

        # 4. Verify which are actionable and prepare input for linear search
        logger.info("[Stage 4.4] Selecting actionable numerical features and gathering "
                    "their bounds for the LP solver.")
        indices_to_modify = [i for i in self.sample_state.shap_coeffs.keys() if i not in self.priorities_state.non_actionable_indices]
        coeff_to_linear_search = [self.sample_state.shap_coeffs[key] for key in indices_to_modify]
        bounds_for_linear_search = [self.priorities_state.bounds[key] for key in indices_to_modify]
        logger.info("[Stage 4.4] %d actionable numerical features: %s",
                    len(indices_to_modify), indices_to_modify)
        logger.debug("[Stage 4.4] LP coefficients=%s | bounds=%s",
                     coeff_to_linear_search, bounds_for_linear_search)


        # 5. For each categorical combination, solve the linear programming problem to find at least one solution
        # TODO: Optimize by removing for loop and vectorize if possible
        logger.info("[Stage 4.5] Solving an LP per categorical combination "
                    "(coefficients . x = adjusted_target, bounded). The first feasible "
                    "x becomes the warm-start for SLSQP later on.")
        combo_and_initial_solutions = {}
        shap_dict = self.sample_state.shapley_values
        cat_combinations = self.sample_state.all_combinations
        priorities_for_search = self.sample_state.limited_priorities
        shap_coefficients = self.sample_state.shap_coeffs
        basic_prediction = self.basic_prediction
        # (true model gap, linearised gap) at each feasible warm-start x0, used
        # to diagnose warm-start health / surrogate mismatch.
        warm_gaps = []

        for combination_id, temp_target in target_for_combo.items():
            logger.info("[Stage 4.5] Combo %d/%d -> LP target=%.4f",
                        combination_id + 1, len(target_for_combo), float(temp_target))
            temp_combo = cat_combinations[combination_id]
            logger.debug("[Stage 4.5] Combo %d categorical assignment: %s",
                         combination_id, temp_combo)
            dummy_x = self.sample_state.sample.copy()

            for idx, val in temp_combo.items():
                dummy_x[idx] = val

            linear_solution = MINLSearchExplainer.solve_linear_constraint_lp(coeff_to_linear_search, temp_target, bounds_for_linear_search, method='auto', tolerance=self.epsilon)
            logger.info("[Stage 4.5] Combo %d LP %s (method=%s).",
                        combination_id,
                        "feasible" if linear_solution["success"] else "infeasible",
                        linear_solution.get("method_used"))
            solution = linear_solution['solution']
            logger.debug("[Stage 4.5] Combo %d LP solution=%s",
                         combination_id, solution)
            if solution is not None:
                solutions_indices = {key: value for key, value in zip(indices_to_modify, solution)}
                combo_and_initial_solutions[combination_id] = {'initial_solution': solutions_indices, 'categorical_combo': temp_combo}
                for key, value in solutions_indices.items():
                    dummy_x[key] = value
                logger.debug("[Stage 4.5] Combo %d candidate vector: %s",
                             combination_id, dummy_x)
                check_value = self.constraint_function(dummy_x, shap_dict, self.sample_state.sample, self.sample_state.target_exemplar, priorities_for_search, basic_prediction=self.basic_prediction)
                assert abs(check_value - self.target) <= self.epsilon, \
                    f"Constraint function value {check_value} exceeds tolerance {self.epsilon} from target {self.target}"
                linear_gap = abs(float(check_value) - float(self.target))
                logger.info("[Stage 4.5] Combo %d sanity check: linearised h(x)=%.4f "
                            "(target=%.4f, |gap|=%.4f, epsilon=%.4f).",
                            combination_id, float(check_value),
                            float(self.target), linear_gap, float(self.epsilon))
                # Surrogate mismatch: true model prediction at the warm-start x0.
                ws_model_pred = float(
                    np.asarray(self.model_pred([dummy_x])).reshape(-1)[0])
                ws_model_gap = abs(ws_model_pred - float(self.target))
                warm_gaps.append((ws_model_gap, linear_gap))
                logger.info("[Stage 4.5] Combo %d warm-start true model_pred=%.4f "
                            "(|model gap|=%.4f vs linear gap=%.4f).",
                            combination_id, ws_model_pred, ws_model_gap, linear_gap)

        logger.info("[Stage 4.6] %d/%d categorical combination(s) yielded an initial "
                    "feasible solution.",
                    len(combo_and_initial_solutions), len(target_for_combo))

        # Capture warm-start health once per find_counterfactuals run (first pass).
        if self._warm_start_info is None:
            best = min(warm_gaps, key=lambda t: t[0]) if warm_gaps else (None, None)
            self._warm_start_info = {
                "total_combos": int(len(target_for_combo)),
                "feasible_combos": int(len(combo_and_initial_solutions)),
                "best_warmstart_model_gap": (float(best[0]) if best[0] is not None else None),
                "best_warmstart_linear_gap": (float(best[1]) if best[1] is not None else None),
            }
            logger.info("[Stage 4.6] Warm-start health: %d/%d combos feasible; best "
                        "warm-start model |gap|=%s | linear |gap|=%s.",
                        self._warm_start_info["feasible_combos"],
                        self._warm_start_info["total_combos"],
                        self._warm_start_info["best_warmstart_model_gap"],
                        self._warm_start_info["best_warmstart_linear_gap"])

        # 6. If no feasible solutions were found, analyze why and suggest bounds adjustments
        # TODO: It worked fine when there were no categorical fatures - I need to rethink this part
        # if len(combo_and_initial_solutions) == 0:
        #     # Try to find intelligent bounds adjustments
        #     bounds_adjustment = self.find_required_bounds_adjustments(
        #         cat_combinations, shap_dict, priorities_for_search, 
        #         basic_prediction, target_for_combo, shap_coefficients
        #     )
            
        #     if bounds_adjustment['feasible_with_shifts']:
        #         logger.warning("No feasible solutions found with current bounds, but solutions possible with adjusted bounds:")
        #         for feature_idx, adjustment in bounds_adjustment['required_shifts'].items():
        #             logger.warning(f"Feature {feature_idx}: {adjustment['justification']}")
        #             logger.warning(f"  Current bounds: {adjustment['original_bounds']}")
        #             logger.warning(f"  Suggested bounds: {adjustment['suggested_bounds']}")
                
        #         raise ValueError(
        #             f"No feasible initial solutions found for any categorical combination. "
        #             f"However, solutions may be possible with adjusted bounds. "
        #             f"Consider adjusting bounds for features: {list(bounds_adjustment['required_shifts'].keys())}"
        #         )
        #     else:
        #         raise ValueError("No feasible initial solutions found for any categorical combination, even with bounds adjustments.")

        return combo_and_initial_solutions

    #################################################################
    # Functions for optimisation
    #################################################################

    # constraint function
    def constraint_function(self, x, shap_dict, sample, target_exemplar, priorities_for_search, basic_prediction):
        """
        Constraint function based on SHAP values and feature changes. 
        
        This function estimates the model prediction for input x using linear approximation
        based on Shapley values. It's used as a constraint during optimization to ensure
        the counterfactual achieves the target prediction.
        
        The approximation works by:
        1. Starting with the basic prediction for the original sample
        2. Adding linear contributions from numerical features based on their Shapley values
        3. Adding discrete contributions from categorical features based on their Shapley values
        
        Args:
            x: Current feature values array/list
            shap_dict: Dictionary containing SHAP values for numerical and categorical features
            sample: Original sample feature values array/list
            target_exemplar: Target exemplar feature values array/list
            priorities_for_search: Modified priorities dictionary
            basic_prediction: Model prediction for the original sample
            
        Returns:
            float: Estimated prediction value based on Shapley value approximation
        """
        num_shap=shap_dict['numerical']
        cat_shap=shap_dict['categorical']

        result = basic_prediction

        for key in priorities_for_search['numerical'].keys():
            # Use pre-calculated coefficient from shap_coeffs instead of recalculating.
            # ``x`` is always the full-length feature vector, so index by the
            # feature's actual column index (numerical indices may be
            # non-contiguous when categorical one-hot columns are present).
            temp_unit_phi = self.sample_state.shap_coeffs[key]

            result += temp_unit_phi * (x[key] - sample[key])

        for key in priorities_for_search['categorical'].keys():
            feature_indices = key
            shap_value = cat_shap[feature_indices]
            
            # Extract current values from x
            current_values = tuple(float(x[idx]) for idx in feature_indices) #tuple(float(x[len(priorities_for_search['numerical']) + i + j]) for j in range(len(feature_indices)))
            
            sample_values = tuple(float(sample[idx]) for idx in feature_indices)
            target_exemplar_values = tuple(float(target_exemplar[idx]) for idx in feature_indices)

            if current_values == sample_values:
                result += 0.0
            elif current_values == target_exemplar_values:
                result += shap_value
            else:
                raise ValueError("Invalid categorical combination encountered for indices {}: {}".format(feature_indices, current_values))

        return result

    #objective function
    def calculate_total_weight(self, values):
        """
        Calculate total weight (cost) based on priorities dictionary and input values list.
        
        This function computes the total cost of changing features from the original sample
        to the proposed counterfactual values. It validates that all values are within 
        allowed ranges for numerical features and acceptable categorical combinations.
        
        The weight calculation follows the priority functions defined in self.priorities:
        - For numerical features: applies the weight function to the feature value
        - For categorical features: uses predefined weights for specific combinations
        - For non-actionable features (function=None): contributes 0 to total weight
        
        Args:
            values: List of values where each element corresponds to its feature index
        
        Returns:
            float: Total calculated weight (cost) for the proposed feature changes
        
        Raises:
            ValueError: If any value is outside allowed range or invalid categorical combination
        """
        total_weight = 0
        
        # Process numerical features
        for feature_id, feature_config in self.priorities_state.numerical_priorities.items():
            if feature_id >= len(values):
                raise ValueError(f"Not enough values provided. Missing value for index {feature_id}")
            
            current_value = values[feature_id]
            
            # Check if feature is non-actionable (function=None)
            if isinstance(feature_config, dict):
                function = feature_config.get('function')
                if function is None:
                    # Non-actionable feature, contributes 0 to total weight
                    # Should be fixed at min=max=sample value
                    pass
                else:
                    # Actionable feature: validate value is within min/max range
                    min_val = feature_config['min']
                    max_val = feature_config['max']
                    
                    if current_value < min_val or current_value > max_val:
                        raise ValueError(f"Value {current_value} at index {feature_id} is outside allowed range [{min_val}, {max_val}]")
                    allowed_intervals = feature_config.get('allowed_intervals')
                    if allowed_intervals and not self._in_allowed_intervals(float(current_value), allowed_intervals):
                        raise ValueError(
                            f"Value {current_value} at index {feature_id} is in a zero-priority gap; "
                            f"allowed intervals: {allowed_intervals}"
                        )
                    
                    # Apply the weight function
                    weight = function(current_value)
                    total_weight += weight
            else:
                # Old format fallback (feature_config == 0 or None)
                # Non-actionable, contributes 0 to total weight
                pass
        
        # Process categorical features
        for feature_indices, categorical_weights in self.priorities_state.categorical_priorities.items():
            # Extract values for this categorical group using their indices
            group_values = []
            for idx in feature_indices:
                if idx >= len(values):
                    raise ValueError(f"Not enough values provided. Missing value for index {idx}")
                group_values.append(float(values[idx]))
            
            # Convert to tuple for dictionary lookup
            value_tuple = tuple(group_values)
            
            # Validate categorical combination exists in allowed combinations
            if value_tuple not in categorical_weights:
                allowed_combinations = list(categorical_weights.keys())
                raise ValueError(f"Invalid categorical combination: {value_tuple} for features {feature_indices}. "
                            f"Allowed combinations: {allowed_combinations}")
            
            # Add weight for this categorical combination
            total_weight += categorical_weights[value_tuple]
        
        return total_weight


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
        logger.info("Goal: pick an anchor whose model prediction is close to the "
                    "requested target. It anchors the linear (Shapley) approximation "
                    "used in later stages.")

        try:
            self.find_closest_elem()
            self.exemplar_source = "dataset_priority_filtered"
            self.exemplar_pred_distance = self._exemplar_pred_distance()
            logger.info("Target exemplar via '%s': |pred - target|=%.4f "
                        "(target=%.4f).",
                        self.exemplar_source,
                        self.exemplar_pred_distance,
                        float(self.target))
            return
        except Exception as primary_exc:
            logger.warning("[Stage 1] Primary exemplar selection "
                           "('dataset_priority_filtered') failed: %s", primary_exc)
            try:
                self._generate_exemplar_in_range(
                    max_iterations=int(getattr(
                        self, "_fallback_random_max_iterations", 10000)),
                )
                self.exemplar_source = "random_in_range"
                self.exemplar_pred_distance = self._exemplar_pred_distance()
                logger.warning("[Stage 1] Randomly generated in-range exemplar "
                               "(|pred - target|=%.4f).",
                               self.exemplar_pred_distance)
                return
            except Exception as rnd_exc:
                self.exemplar_source = "none"
                logger.error("[Stage 1] Could not obtain any exemplar (dataset or "
                             "random in-range): %s", rnd_exc)
                raise

    def _exemplar_pred_distance(self) -> float:
        """``|model_pred(target_exemplar) - target|`` for the current exemplar."""
        pred = float(
            np.asarray(self.model_pred([self.sample_state.target_exemplar])).reshape(-1)[0])
        return abs(pred - float(self.target))

    def _sample_from_allowed_region(self, cfg: dict) -> float:
        """Draw a uniform value from a numerical feature's allowed region."""
        intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None
        if not intervals:
            lo = cfg.get("min")
            hi = cfg.get("max")
            if lo is None or hi is None:
                raise Exception("Feature has no allowed region to sample from.")
            return float(np.random.uniform(float(lo), float(hi)))
        lengths = np.array(
            [max(0.0, float(hi) - float(lo)) for lo, hi in intervals], dtype=float)
        total = float(lengths.sum())
        if total <= 0.0:
            lo, _hi = intervals[int(np.random.randint(len(intervals)))]
            return float(lo)
        pick = intervals[int(np.random.choice(len(intervals), p=lengths / total))]
        return float(np.random.uniform(float(pick[0]), float(pick[1])))

    def _sample_candidate_in_allowed_region(self) -> np.ndarray:
        """Assemble one random candidate inside the allowed region.

        Actionable numerical features are drawn uniformly from their allowed
        interval(s); actionable categorical groups pick uniformly among allowed
        combinations (priority weights are ignored on purpose); non-actionable
        features stay frozen at the sample value.
        """
        sample = np.asarray(self.sample_state.sample, dtype=float)
        cand = sample.copy()
        for idx, cfg in self.priorities_state.numerical_priorities.items():
            if not isinstance(cfg, dict) or cfg.get("function") is None:
                cand[idx] = float(sample[idx])  # non-actionable: frozen
                continue
            cand[idx] = self._sample_from_allowed_region(cfg)
        for group, mapping in self.priorities_state.categorical_priorities.items():
            allowed = [combo for combo, weight in mapping.items()
                       if weight is not None and float(weight) > 0.0]
            if not allowed:
                raise Exception(f"Categorical group {group} has no allowed "
                                f"combinations to sample from.")
            combo = allowed[int(np.random.randint(len(allowed)))]
            for j, idx in enumerate(group):
                cand[idx] = float(combo[j])
        return cand

    def _generate_exemplar_in_range(
        self, max_iterations: int = 10000, random_seed=None,
    ) -> float:
        """Randomly generate an in-range exemplar near the target.

        Samples the allowed region (see :meth:`_sample_candidate_in_allowed_region`)
        and accepts the first candidate whose prediction lands within ``epsilon``
        of the target, without regard to its priority score. This is *not* the
        random-search comparison method - it only produces a reachable anchor for
        MINLP; the subsequent optimisation is what maximises the preference score.
        Returns ``|prediction - target|`` of the accepted point.
        """
        logger.info("[Stage 1 - fallback] Randomly generating an in-range exemplar: "
                    "sampling the allowed region for a point within epsilon=%.4f of "
                    "target (max_iterations=%d).", float(self.epsilon), int(max_iterations))
        if random_seed is not None:
            np.random.seed(int(random_seed))

        for i in range(int(max_iterations)):
            cand = self._sample_candidate_in_allowed_region()
            pred = float(np.asarray(self.model_pred(cand.reshape(1, -1))).reshape(-1)[0])
            if abs(pred - float(self.target)) <= float(self.epsilon):
                self.sample_state.target_exemplar = cand
                logger.info("[Stage 1 - fallback] In-range exemplar found at iteration "
                            "%d (|pred - target|=%.4f).",
                            i + 1, abs(pred - float(self.target)))
                return abs(pred - float(self.target))

        raise Exception(f"Could not randomly generate an in-range point within "
                        f"epsilon={self.epsilon} of target after {max_iterations} "
                        f"iterations.")

    def probe_allowed_region_feasibility(
        self, max_iterations: int = 5000, random_seed=None,
    ) -> dict:
        """Measure whether the allowed region can reach the target at all.

        Randomly samples the allowed region (ignoring preference, exactly like
        :meth:`_generate_exemplar_in_range`) and reports whether any point lands
        within ``epsilon`` of the target and the smallest ``|pred - target|``
        seen. This separates *feasibility* (region cannot reach target) from
        *convergence* (region can, but MINLP's optimiser did not) failures.

        Does not touch ``sample_state.target_exemplar``. Runs Stage 0 first so
        the allowed intervals exist.
        """
        self._derive_bounds_and_intervals_from_priorities()
        if random_seed is not None:
            np.random.seed(int(random_seed))

        min_dist = float("inf")
        found_iter = None
        for i in range(int(max_iterations)):
            cand = self._sample_candidate_in_allowed_region()
            pred = float(np.asarray(self.model_pred(cand.reshape(1, -1))).reshape(-1)[0])
            dist = abs(pred - float(self.target))
            if dist < min_dist:
                min_dist = dist
            if dist <= float(self.epsilon):
                found_iter = i + 1
                break
        return {
            "feasible_within_epsilon": found_iter is not None,
            "min_pred_distance": float(min_dist),
            "iterations_run": int(found_iter if found_iter is not None else max_iterations),
        }

    def _find_exemplar_random_fallback(
        self, max_iterations: int = 10000, random_seed=None,
    ) -> float:
        """Sample points from the allowed region; return the first near the target.

        Each actionable numerical feature is drawn uniformly from its allowed
        interval(s); each actionable categorical group picks uniformly among its
        allowed combinations (priority weights are ignored on purpose).
        Non-actionable features are frozen at the sample value. The first
        candidate whose prediction lands within ``epsilon`` of the target is
        accepted, without regard to its priority score. Returns
        ``|prediction - target|`` of the accepted point.
        """
        logger.info("[Stage 1 - fallback] Random-search exemplar: sampling the allowed "
                    "region for a point within epsilon=%.4f of target "
                    "(max_iterations=%d).", float(self.epsilon), int(max_iterations))
        if random_seed is not None:
            np.random.seed(int(random_seed))

        sample = np.asarray(self.sample_state.sample, dtype=float)
        num_priorities = self.priorities_state.numerical_priorities
        cat_priorities = self.priorities_state.categorical_priorities

        for i in range(int(max_iterations)):
            cand = sample.copy()
            for idx, cfg in num_priorities.items():
                if not isinstance(cfg, dict) or cfg.get("function") is None:
                    cand[idx] = float(sample[idx])  # non-actionable: frozen
                    continue
                cand[idx] = self._sample_from_allowed_region(cfg)
            for group, mapping in cat_priorities.items():
                allowed = [combo for combo, weight in mapping.items()
                           if weight is not None and float(weight) > 0.0]
                if not allowed:
                    raise Exception(f"Categorical group {group} has no allowed "
                                    f"combinations to sample from.")
                combo = allowed[int(np.random.randint(len(allowed)))]
                for j, idx in enumerate(group):
                    cand[idx] = float(combo[j])
            pred = float(np.asarray(self.model_pred(cand.reshape(1, -1))).reshape(-1)[0])
            if abs(pred - float(self.target)) <= float(self.epsilon):
                self.sample_state.target_exemplar = cand
                logger.info("[Stage 1 - fallback] Random-search found a point at "
                            "iteration %d (|pred - target|=%.4f).",
                            i + 1, abs(pred - float(self.target)))
                return abs(pred - float(self.target))

        raise Exception(f"Random-search fallback could not find a point within "
                        f"epsilon={self.epsilon} of target after {max_iterations} "
                        f"iterations.")

    def _stage2_log_bounds(self):
        """Surface the numerical bounds used by both LP and SLSQP."""
        logger.info("--- STAGE 2/6: Collect numerical-feature bounds from priorities ---")
        logger.info("These (min, max) pairs constrain how far each numerical feature is "
                    "allowed to move during the LP and SLSQP searches.")
        logger.info("Bounds for numerical features: %s", self.priorities_state.bounds)
        logger.info("Non-actionable feature indices (frozen at sample values): %s",
                    self.priorities_state.non_actionable_indices)

    def _run_one_pass(self, shap_approx, num_samples):
        """Execute stages 3, 4 and 5 once for the current ``sample_state.sample``.

        Returns the list of candidate counterfactuals (one per surviving
        categorical combination). Stage 6 selection is delegated to
        :meth:`_select_best_candidate` so the iterative loop in
        :meth:`find_counterfactuals` can reuse both helpers cleanly.
        """
        # Stage 3: Shapley values for the (possibly updated) current sample.
        logger.info("--- STAGE 3/6: Compute Shapley values (sample vs. target exemplar) ---")
        logger.info("Shapley values quantify each feature's contribution to the gap "
                    "between the prediction on the sample and on the target exemplar. "
                    "Numerical features get one value each; one-hot categorical groups "
                    "are consolidated into a single value per group.")
        self.calc_shapley(self.sample_state.sample,
                          use_approximation=shap_approx,
                          num_samples=num_samples)
        logger.info("Shapley values (numerical): %s",
                    self.sample_state.shapley_values.get('numerical'))
        logger.info("Shapley values (categorical groups): %s",
                    self.sample_state.shapley_values.get('categorical'))

        # Stage 4: warm-start LP per surviving categorical combination.
        logger.info("--- STAGE 4/6: Build initial feasible solutions per categorical combo ---")
        logger.info("For each surviving categorical combination, solve a linear program "
                    "(in the Shapley-linearised model) to find numerical values that "
                    "land within +/- epsilon of the target. These become warm starts "
                    "for the nonlinear SLSQP optimisation in stage 5.")
        init_vals_per_combo = self.confirm_existence_of_solution_for_combo()
        logger.info("Initial values per combination: %s", init_vals_per_combo)
        logger.info("Limited priorities used for search: %s",
                    self.sample_state.limited_priorities)

        # Stage 5: SLSQP per combo. Wrappers and bounds match the original flow.
        logger.info("--- STAGE 5/6: SLSQP optimisation per categorical combination ---")
        logger.info("Minimise the priority cost subject to the Shapley-linear constraint "
                    "h(x) in [target-epsilon, target+epsilon]. One run per categorical "
                    "combination; without categorical features the loop runs once.")
        bounds = self.priorities_state.bounds
        counterfactuals = []
        for i, values in init_vals_per_combo.items():
            logger.info("[combo %d/%d] Preparing input from initial LP solution and "
                        "categorical assignment...", i + 1, len(init_vals_per_combo))
            prepared_input = self.sample_state.sample.copy()
            initial_numerical = values['initial_solution']
            cat_combo = values['categorical_combo']
            for idx, val in cat_combo.items():
                prepared_input[idx] = val
            for idx, val in initial_numerical.items():
                prepared_input[idx] = val
            logger.debug("Prepared input: %s", prepared_input)

            def constraint_wrapper(x, shap_dict, sample, target_exemplar, priorities_for_search, basic_prediction, ready_input):
                # ``x`` is the compact SLSQP vector (one entry per numerical
                # feature, ordered like ``shap_dict['numerical']``); map each
                # position back onto its original column in the full vector.
                for j, idx in enumerate(shap_dict['numerical'].keys()):
                    ready_input[idx] = x[j]

                return self.constraint_function(ready_input, shap_dict, sample, target_exemplar, priorities_for_search, basic_prediction)

            def objective_wrapper(x, priorities_for_search, ready_input):
                for j, idx in enumerate(self.sample_state.shapley_values['numerical'].keys()):
                    ready_input[idx] = x[j]
                try:
                    return self.calculate_total_weight(ready_input)
                except ValueError:
                    return -1e12

            constraint_fun = lambda x: constraint_wrapper(x, self.sample_state.shapley_values, self.sample_state.sample, self.sample_state.target_exemplar, self.sample_state.limited_priorities, basic_prediction=self.basic_prediction, ready_input=prepared_input.copy())
            objective_fun = lambda x: -objective_wrapper(x, self.sample_state.limited_priorities, prepared_input.copy())

            # Lower bound constraint: h(x) >= target - epsilon
            def lower_constraint(x):
                return constraint_fun(x) - (self.target - self.epsilon)

            # Upper bound constraint: h(x) <= target + epsilon
            def upper_constraint(x):
                return (self.target + self.epsilon) - constraint_fun(x)

            constraints = [
                {'type': 'ineq', 'fun': lower_constraint},
                {'type': 'ineq', 'fun': upper_constraint},
            ]
            numerical_indices = list(self.priorities_state.numerical_priorities.keys())
            x0 = np.array([prepared_input[idx] for idx in numerical_indices])
            bounds_list = [bounds[idx] for idx in numerical_indices]
            for j, feature_idx in enumerate(numerical_indices):
                cfg = self.priorities_state.numerical_priorities.get(feature_idx, {})
                allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None
                if allowed_intervals:
                    x0[j] = self._project_to_allowed_intervals(float(x0[j]), allowed_intervals)
                    constraints.append({
                        'type': 'ineq',
                        'fun': (
                            lambda x, _j=j, _intervals=allowed_intervals:
                            self._allowed_interval_constraint_value(float(x[_j]), _intervals)
                        ),
                    })
            logger.info("[combo %d/%d] Initial numerical x0=%s",
                        i + 1, len(init_vals_per_combo), x0)
            logger.debug("[combo %d/%d] Constraints (bounded): h(x) in [%.4f, %.4f]",
                         i + 1, len(init_vals_per_combo),
                         self.target - self.epsilon, self.target + self.epsilon)
            logger.debug("[combo %d/%d] Bounds list passed to SLSQP: %s",
                         i + 1, len(init_vals_per_combo), bounds_list)
            logger.info("[combo %d/%d] Running SLSQP (maximise priority weight subject "
                        "to Shapley-linear constraint)...",
                        i + 1, len(init_vals_per_combo))
            result = minimize(
                objective_fun,
                x0,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraints,
            )

            complete_solution = prepared_input.copy()
            for j, idx in enumerate(self.sample_state.shapley_values['numerical'].keys()):
                complete_solution[idx] = result.x[j]
            counterfactuals.append(complete_solution)
            model_prediction = float(self.model_pred([complete_solution])[0])
            abs_gap = abs(model_prediction - float(self.target))
            logger.info("[combo %d/%d] SLSQP done: success=%s | iterations=%s | "
                        "model_pred(cf)=%.4f | constraint h(x)=%.4f | "
                        "objective(weight)=%.4f",
                        i + 1, len(init_vals_per_combo),
                        getattr(result, "success", None),
                        getattr(result, "nit", "?"),
                        model_prediction,
                        float(constraint_fun(result.x)),
                        float(-result.fun))
            iter_no = getattr(self, "_workflow_iteration", None)
            if iter_no is None:
                self._workflow_log(
                    "4) Proposition combo=%d: prediction=%.6f | target=%.6f | |gap|=%.6f",
                    i + 1, model_prediction, float(self.target), abs_gap,
                )
            else:
                self._workflow_log(
                    "4) Iteration %d proposition combo=%d: prediction=%.6f | target=%.6f | |gap|=%.6f",
                    int(iter_no), i + 1, model_prediction, float(self.target), abs_gap,
                )
            logger.debug("[combo %d/%d] Optimal x=%s | message=%s",
                         i + 1, len(init_vals_per_combo),
                         result.x, getattr(result, "message", ""))
            logger.debug("-----")

        return counterfactuals

    def _priority_benefit(self, cf):
        """``calculate_total_weight(cf)`` or ``None`` when ``cf`` is out of range.

        ``calculate_total_weight`` is the *priority benefit* that the search
        maximises (higher is better, same quantity as
        ``priority_methods.methods.compute_priority_score``). It raises for
        values outside the allowed range / inside a zero-priority gap, which
        SLSQP can still produce, so callers get ``None`` instead of an
        exception.
        """
        try:
            return float(self.calculate_total_weight(cf))
        except ValueError as exc:
            logger.debug("Priority benefit unavailable for candidate: %s", exc)
            return None

    def _select_best_candidate(self, counterfactuals):
        """Stage 6: pick the candidate with the highest priority benefit.

        ``calculate_total_weight`` is maximised by the SLSQP objective (which
        minimises its negation), so selection across categorical combinations
        must pick the *largest* value. With no categorical features there is
        exactly one candidate so this just returns it. Raises
        :class:`RuntimeError` if no candidates are available so the iterative
        loop can mark the iteration as failed.
        """
        logger.info("--- STAGE 6/6: Pick best counterfactual ---")
        logger.info("Among %d candidate(s), pick the one with the highest priority "
                    "benefit (calculate_total_weight). With no categorical features "
                    "there is exactly one candidate.", len(counterfactuals))
        if not counterfactuals:
            raise RuntimeError("No counterfactual candidates produced for selection.")
        if len(counterfactuals) > 1:
            best_counterfactual = None
            best_benefit = float('-inf')
            for cf in counterfactuals:
                benefit = self._priority_benefit(cf)
                logger.debug("Candidate priority benefit=%s", benefit)
                if benefit is not None and benefit > best_benefit:
                    best_benefit = benefit
                    best_counterfactual = cf
            if best_counterfactual is None:
                logger.warning("No candidate had an evaluable priority benefit; "
                               "falling back to the first candidate.")
                return counterfactuals[0]
            logger.info("Selected candidate priority benefit=%.4f", best_benefit)
            return best_counterfactual
        return counterfactuals[0]

    def _evaluate_candidate(self, cf):
        """Score ``cf`` against the real model and the Shapley surrogate.

        Returns a dict with:
          * ``model_pred``: real model output on ``cf``.
          * ``h_x``: surrogate prediction (Shapley-linear approximation).
            Falls back to NaN if the surrogate cannot be evaluated yet.
          * ``distance``: ``|model_pred - target|``, used to rank candidates
            only while no candidate satisfies the target band.
          * ``priority``: priority benefit (``None`` if not evaluable), used
            to rank candidates that do satisfy the band.
          * ``feasible``: ``distance <= epsilon`` against the real model.
        """
        model_pred = float(self.model_pred([cf])[0])
        try:
            h_x = float(self.constraint_function(
                cf,
                self.sample_state.shapley_values,
                self.sample_state.sample,
                self.sample_state.target_exemplar,
                self.sample_state.limited_priorities,
                basic_prediction=self.basic_prediction,
            ))
        except Exception:
            h_x = float("nan")
        distance = abs(model_pred - float(self.target))
        return {
            "model_pred": model_pred,
            "h_x": h_x,
            "distance": distance,
            "priority": self._priority_benefit(cf),
            "feasible": distance <= float(self.epsilon),
        }

    def find_counterfactuals(self, shap_approx=False, num_samples=200,
                             max_iterations=10, patience=5,
                             return_when_fails=True,
                             fallback_random_max_iterations=10000):
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
        logger.info("MINLP COUNTERFACTUAL SEARCH | target=%.4f | epsilon=%.4f | "
                    "max_iterations=%d | patience=%d | shap_approx=%s | "
                    "num_samples=%d | return_when_fails=%s",
                    float(self.target), float(self.epsilon),
                    int(max_iterations), int(patience),
                    bool(shap_approx), int(num_samples),
                    bool(return_when_fails))
        logger.info("=" * 78)

        self.exemplar_source = None
        self.exemplar_pred_distance = None
        self._warm_start_info = None
        self._last_search_exception = None
        self._fallback_random_max_iterations = int(fallback_random_max_iterations)

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
                anchor_priority if anchor_priority is not None else float("-inf"))
            best_feasible_distance = anchor_distance
            best_feasible_source = "anchor"
            logger.info("Initial exemplar is already a valid fallback CF: "
                        "model_pred=%.4f | distance=%.4f | priority=%s. It is only "
                        "kept as the incumbent until an SLSQP candidate beats its "
                        "priority benefit.",
                        anchor_pred, anchor_distance, anchor_priority)
        no_progress = 0
        history = []
        stop_reason = "max_iterations"

        for iteration in range(int(max_iterations)):
            logger.info("############ REFINEMENT ITERATION %d/%d ############",
                        iteration + 1, int(max_iterations))
            self._workflow_iteration = iteration + 1
            try:
                candidates = self._run_one_pass(shap_approx, num_samples)
                cf = self._select_best_candidate(candidates)
            except Exception as exc:
                logger.warning("Iteration %d failed during search pass: %s",
                               iteration + 1, exc)
                stop_reason = "search_failed"
                self._last_search_exception = str(exc)
                break

            eval_info = self._evaluate_candidate(cf)
            priority = eval_info["priority"]
            improved = False

            if eval_info["feasible"]:
                # Inside the band: rank on the priority benefit only.
                if best_feasible is None or (
                    priority is not None and priority > best_feasible_priority
                ):
                    best_feasible = list(cf)
                    best_feasible_priority = (
                        priority if priority is not None else float("-inf"))
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
                if best_feasible_priority != float("-inf") else "n/a"
            )
            best_distance_so_far = (
                f"{best_feasible_distance:.4f}" if best_feasible is not None
                else (f"{best_infeasible_distance:.4f}"
                      if best_infeasible_distance != float("inf") else "n/a")
            )
            logger.info("[Iter %d] model_pred(cf)=%.4f | h(x)=%.4f | distance=%.4f | "
                        "feasible=%s | priority=%s | best_priority=%s | "
                        "best_distance=%s | improved_vs_best=%s",
                        iteration + 1,
                        eval_info["model_pred"], eval_info["h_x"],
                        eval_info["distance"], eval_info["feasible"],
                        f"{priority:.4f}" if priority is not None else "n/a",
                        best_priority_so_far, best_distance_so_far, improved)

            if improved:
                no_progress = 0
            else:
                no_progress += 1
                logger.info("[Iter %d] No improvement vs incumbent (%d/%d).",
                            iteration + 1, no_progress, int(patience))

            history.append({
                "iteration": iteration + 1,
                "model_pred": eval_info["model_pred"],
                "h_x": eval_info["h_x"],
                "distance": eval_info["distance"],
                "feasible": eval_info["feasible"],
                "priority": priority,
                "improved_vs_best": improved,
            })

            if no_progress >= int(patience):
                logger.info("Stopping: %d consecutive iterations without improving "
                            "the %s incumbent.", int(patience),
                            "priority" if best_feasible is not None else "distance")
                stop_reason = ("priority_stagnation" if best_feasible is not None
                               else "patience_exhausted")
                break

            # Always advance to the latest CF, even when it did not improve;
            # the next iteration re-linearises against the same exemplar.
            self.sample_state.sample = list(cf)
            logger.info("[Iter %d] Advancing: next iteration's sample = current CF.",
                        iteration + 1)

        # Restore the original sample so explainer state stays consistent
        # for any caller that inspects sample_state after the search.
        self.sample_state.sample = original_sample
        self._workflow_iteration = None

        reached = best_feasible is not None
        if reached:
            best_cf = best_feasible
            best_distance = best_feasible_distance
            best_priority = (best_feasible_priority
                             if best_feasible_priority != float("-inf") else None)
            cf_source = best_feasible_source
            if stop_reason == "max_iterations":
                stop_reason = "target_reached"
        else:
            best_cf = best_infeasible
            best_distance = best_infeasible_distance
            best_priority = (self._priority_benefit(best_cf)
                             if best_cf is not None else None)
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
        }

        logger.info("=" * 78)
        logger.info("MINLP SEARCH DONE | reached_target=%s | stop_reason=%s | "
                    "iterations=%d | best_distance=%.4f | cf_source=%s | "
                    "priority=%s | anchor_priority=%s | gain_vs_anchor=%s",
                    reached, stop_reason, len(history),
                    best_distance if best_distance != float("inf") else float("nan"),
                    cf_source, best_priority, anchor_priority,
                    priority_gain_vs_anchor)
        if cf_source == "anchor":
            logger.warning("Returned CF is the Stage 1 anchor exemplar (source=%s): no "
                           "SLSQP candidate beat its priority benefit (%s). This run "
                           "carries no optimisation gain.",
                           self.exemplar_source, anchor_priority)
        logger.info("=" * 78)

        if reached:
            return best_cf
        if return_when_fails:
            if best_cf is None:
                logger.warning("No candidate was produced; returning None despite "
                               "return_when_fails=True.")
                return None
            logger.warning("Returning best CF although it did NOT reach target "
                           "(distance=%.4f > epsilon=%.4f). See "
                           "self.last_search_result for full status.",
                           best_distance, float(self.epsilon))
            return best_cf
        logger.warning("Target not reached and return_when_fails=False; returning None.")
        return None

    def find_counterfactuals_for_binary(self, target_class=1, threshold=0.5, 
                                       expected_counterfactuals=10, max_iterations=100,
                                       shap_approx=False, num_samples=200, return_top_n=None):
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
        logger.info(f"Starting MINLP-based binary classification counterfactual search...")
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
            logger.info(f"Target exemplar prediction: {self.model_pred([self.sample_state.target_exemplar])[0]:.4f}")
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
        self.calc_shapley(self.sample_state.sample, use_approximation=shap_approx, num_samples=num_samples)
        logger.info(f'SHAP values computed: {len(self.sample_state.shapley_values["numerical"])} numerical, '
                   f'{len(self.sample_state.shapley_values["categorical"])} categorical features')

        ###########
        # 4. Create initial values per categorical combination
        try:
            init_vals_per_combo = self.confirm_existence_of_solution_for_combo()
        except (ValueError, AssertionError) as e:
            logger.warning(f"Could not confirm solution existence: {e}")
            logger.info("Attempting to proceed with optimizations anyway...")
            init_vals_per_combo = {}
        
        if not init_vals_per_combo:
            logger.info("No feasible combinations found. Generating initial values from sample/target...")
            init_vals_per_combo = {0: {
                'initial_solution': {},
                'categorical_combo': {}
            }}

        priorities_for_search = self.sample_state.limited_priorities if hasattr(self.sample_state, 'limited_priorities') else self.priorities_state.priorities
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
            
            if 'initial_solution' in values and values['initial_solution']:
                initial_numerical = values['initial_solution']
                cat_combo = values.get('categorical_combo', {})
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
            
            def objective_wrapper(x):
                temp_input = prepared_input.copy()
                for i, idx in enumerate(numerical_indices):
                    if i < len(x):
                        temp_input[idx] = x[i]
                try:
                    return self.calculate_total_weight(temp_input)
                except ValueError:
                    return -1e12
            
            # For binary classification, use direct model prediction instead of Shapley approximation
            def classification_constraint(x):
                temp_input = prepared_input.copy()
                for i, idx in enumerate(numerical_indices):
                    if i < len(x):
                        temp_input[idx] = x[i]
                pred = self.model_pred([temp_input])[0]
                if target_class == 1:
                    return pred - threshold  # pred > threshold
                else:
                    return threshold - pred  # pred < threshold
            
            constraints = [
                {'type': 'ineq', 'fun': classification_constraint}
            ]
            
            x0 = np.array([prepared_input[idx] for idx in numerical_indices])
            bounds_list = [bounds[idx] for idx in numerical_indices]
            for j, feature_idx in enumerate(numerical_indices):
                cfg = self.priorities_state.numerical_priorities.get(feature_idx, {})
                allowed_intervals = cfg.get("allowed_intervals") if isinstance(cfg, dict) else None
                if allowed_intervals:
                    x0[j] = self._project_to_allowed_intervals(float(x0[j]), allowed_intervals)
                    constraints.append({
                        'type': 'ineq',
                        'fun': (
                            lambda x, _j=j, _intervals=allowed_intervals:
                            self._allowed_interval_constraint_value(float(x[_j]), _intervals)
                        ),
                    })
            
            logger.info(f"Starting optimization for combination {combo_id}...")
            logger.debug(f"Initial values: {x0}")
            logger.debug(f"Bounds: {bounds_list}")
            
            try:
                result = minimize(
                    lambda x: -objective_wrapper(x),  # Maximize weight (minimize negative weight)
                    x0,
                    method='SLSQP',
                    bounds=bounds_list,
                    constraints=constraints,
                    options={'maxiter': max_iterations, 'ftol': 1e-6}
                )
                
                if result.success or not result.success:  # Accept both success and partial convergence
                    complete_solution = prepared_input.copy()
                    for i, idx in enumerate(numerical_indices):
                        complete_solution[idx] = result.x[i]
                    
                    # Verify the solution meets the classification constraint
                    pred = self.model_pred([complete_solution])[0]
                    is_valid = (pred > threshold) if target_class == 1 else (pred < threshold)
                    
                    logger.debug(f"Solution prediction: {pred:.4f}, valid: {is_valid}, message: {result.message}")
                    
                    if is_valid:
                        counterfactuals.append(complete_solution)
                        predictions.append(pred)
                        scores.append(objective_wrapper(result.x))
                        found_counts.append(combo_id + 1)
                        logger.info(f"Found valid counterfactual: pred={pred:.4f}, weight={scores[-1]:.4f}")
                    else:
                        logger.debug(f"Solution doesn't meet classification constraint: pred={pred:.4f}, threshold={threshold}")
            
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