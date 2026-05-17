from explainit.logging_config import logger
from explainit.utils.plot_styles import (apply_style, style_numerical_plot, style_categorical_plot, 
                                        get_line_color, get_bar_color, get_bar_gradient_colors, COLORS)
# logger.info("This is an info message")
# logger.debug("This is a debug message with details")
# logger.warning("This is a warning message")
# logger.error("This is an error message")

import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import factorial
import itertools
import copy
from scipy.optimize import linprog
import warnings
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any

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
                epsilon=0.01):
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

        logger.info("Filtering dataset based on priorities from {} samples...".format(self.dataset.shape[0]))
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
        for group, cat_vals in unwanted_cat_groups:
            data_np = remv_cat(data_np, group, cat_vals)
            if data_np.size == 0:
                raise Exception("There are no elements fulfilling the requirements")
        logger.info(f"After categorical filtering, {data_np.shape[0]} samples remain.")

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
        logger.info(f"After numerical filtering, {data_np.shape[0]} samples remain.")

        return data_np

    ################################################################
    # Finds element in dataset (1)
    ################################################################

    def find_closest_elem(self) -> list:
        """
        Finds the elements in the dataset that are closest to the desired target value +- epsilon.
        Returns the actual samples of all such elements.
        """
        logger.info("Finding the closest element in the dataset based on the specified priorities...")
        filtered_data = self.get_rows_in_priorities()
        pred = self.model_pred(filtered_data)

        min_dist = np.abs(pred - self.target).min()
        if min_dist > self.target_exemplar_epsilon:
            raise ValueError(f"No elements found within the specified epsilon of {self.target_exemplar_epsilon}. Closest distance is {min_dist}.")

        idx_all = np.where(np.abs(pred - self.target) == min_dist)[0]
        assert len(idx_all) == 1, "Expected exactly one target exemplar index, but found multiple."
        self.sample_state.target_exemplar = filtered_data[idx_all[0]]
    
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
            logger.info(f"Computing Shapley for consolidated feature {i}, processing {total_subsets} subsets...")
            
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
            
            logger.info(f"Computing approximate Shapley for consolidated feature {i}, using {num_samples} samples...")
            
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

        logger.info(f"Consolidated {original_length} original features into {total_shapley_features} Shapley calculation units")
        logger.info(f"Numerical features: {len(self.priorities_state.numerical_priorities)}")
        logger.info(f"Categorical groups: {len(cat_groups)}")

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
            temp_unit_phi = num_shap[key]/(self.sample_state.target_exemplar[key] - self.sample_state.sample[key])
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
        self.create_limited_priorities()

        # self.shap_dict = shap_dict
        # 2. Extract targets for linear search for each categorical combination
        self.basic_prediction=self.model_pred([self.sample_state.sample])[0]

        # 3. Prepare targets and coefficients for linear search
        target_for_combo = self.extract_for_linear_search()

        # 4. Verify which are actionable and prepare input for linear search
        indices_to_modify = [i for i in self.sample_state.shap_coeffs.keys() if i not in self.priorities_state.non_actionable_indices]
        coeff_to_linear_search = [self.sample_state.shap_coeffs[key] for key in indices_to_modify]
        bounds_for_linear_search = [self.priorities_state.bounds[key] for key in indices_to_modify]


        # 5. For each categorical combination, solve the linear programming problem to find at least one solution
        # TODO: Optimize by removing for loop and vectorize if possible
        combo_and_initial_solutions = {}
        shap_dict = self.sample_state.shapley_values
        cat_combinations = self.sample_state.all_combinations
        priorities_for_search = self.sample_state.limited_priorities
        shap_coefficients = self.sample_state.shap_coeffs
        basic_prediction = self.basic_prediction

        for combination_id, temp_target in target_for_combo.items():
            logger.info(f'Processing combination_id: {combination_id} with target: {temp_target}')
            temp_combo = cat_combinations[combination_id]
            logger.info(f'temp_combo: {temp_combo}')
            dummy_x = self.sample_state.sample.copy()

            for idx, val in temp_combo.items():
                dummy_x[idx] = val

            linear_solution = MINLSearchExplainer.solve_linear_constraint_lp(coeff_to_linear_search, temp_target, bounds_for_linear_search, method='auto', tolerance=self.epsilon)
            logger.info(f'linear_solution["success"]: {linear_solution["success"]}')
            solution = linear_solution['solution']
            logger.info(f'solution: {solution}')
            if solution is not None:
                solutions_indices = {key: value for key, value in zip(indices_to_modify, solution)}
                combo_and_initial_solutions[combination_id] = {'initial_solution': solutions_indices, 'categorical_combo': temp_combo}
                for key, value in solutions_indices.items():
                    dummy_x[key] = value
                logger.info(f'sample: {self.sample_state.sample}')
                logger.info(f'dummy_x: {dummy_x}')
                assert abs(self.constraint_function(dummy_x, shap_dict, self.sample_state.sample, self.sample_state.target_exemplar, priorities_for_search, basic_prediction=self.basic_prediction) - self.target) <= self.epsilon, \
                    f"Constraint function value {self.constraint_function(dummy_x, shap_dict, self.sample_state.sample, self.sample_state.target_exemplar, priorities_for_search, basic_prediction=self.basic_prediction)} exceeds tolerance {self.epsilon} from target {self.target}"
                logger.info(f'constraint_function: {self.constraint_function(dummy_x, shap_dict, self.sample_state.sample, self.sample_state.target_exemplar, priorities_for_search, basic_prediction=self.basic_prediction)}')

        print('combo_and_initial_solutions: ', combo_and_initial_solutions)

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

        for i, key in enumerate(priorities_for_search['numerical'].keys()):
            # Use pre-calculated coefficient from shap_coeffs instead of recalculating
            temp_unit_phi = self.sample_state.shap_coeffs[key]
            
            result += temp_unit_phi * (x[i] - sample[key])
        
        for i, key in enumerate(priorities_for_search['categorical'].keys()):
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

    def find_counterfactuals(self, shap_approx=False, num_samples=200):
        """
        Find counterfactual explanations using Mixed-Integer Nonlinear Programming (MINLP) approach.
        
        This method uses Shapley values to understand feature importance and then searches for 
        counterfactual examples that achieve the target prediction while minimizing the total 
        cost (weight) based on the priority function.
        
        Args:
            shap_approx: If True, uses sampling-based approximation for Shapley value calculation
            num_samples: Number of samples for Shapley value approximation
            
        Returns:
            list: List of counterfactual examples (modified feature vectors)
        """
        
        ############
        # 1. Finding the target exemplar in the desired target class
        self.find_closest_elem()

        ############
        # 2. Determining bounds for numerical features
        bounds = self.priorities_state.bounds
        logger.info(f"Bounds for numerical features: {bounds}")

        ###########
        # 3. Calculate SHAP values for the sample
        self.calc_shapley(self.sample_state.sample, use_approximation=shap_approx, num_samples=num_samples)
        logger.info(f'SHAP values: {self.sample_state.shapley_values}')

        ###########
        # 4. Create initial values per categorical combination
        init_vals_per_combo = self.confirm_existence_of_solution_for_combo()
        priorities_for_search = self.sample_state.limited_priorities
        logger.info(f"Initial values per combination: {init_vals_per_combo}")
        logger.info(f"Priorities for search: {priorities_for_search}")


        ############
        # 5. For each categorical combination, perform MINLP to find counterfactuals
        # If there are no categorical features, this loop will run only once
        logger.info("Starting MINLP search for counterfactuals...")
        counterfactuals = []
        for i, values in init_vals_per_combo.items():
            logger.info(f"Processing categorical combination {i}...")
            prepared_input = self.sample_state.sample.copy()
            initial_numerical = values['initial_solution']
            cat_combo = values['categorical_combo']
            for idx, val in cat_combo.items():
                prepared_input[idx] = val
            for idx, val in initial_numerical.items():
                prepared_input[idx] = val
            logger.debug("Prepared input: %s", prepared_input)

            def constraint_wrapper(x, shap_dict, sample, target_exemplar, priorities_for_search, basic_prediction, ready_input):
                for idx in shap_dict['numerical'].keys():
                    ready_input[idx] = x[idx]

                return self.constraint_function(ready_input, shap_dict, sample, target_exemplar, priorities_for_search, basic_prediction)
            
            def objective_wrapper(x, priorities_for_search, ready_input):
                for idx in self.sample_state.shapley_values['numerical'].keys():
                    ready_input[idx] = x[idx]
                return self.calculate_total_weight(ready_input)

            constraint_fun = lambda x: constraint_wrapper(x, self.sample_state.shapley_values, self.sample_state.sample, self.sample_state.target_exemplar, self.sample_state.limited_priorities, basic_prediction=self.basic_prediction, ready_input=prepared_input.copy())
            objective_fun = lambda x: -objective_wrapper(x, self.sample_state.limited_priorities, prepared_input.copy())

            # Lower bound constraint: h(x) >= target - epsilon
            def lower_constraint(x):
                return constraint_fun(x) - (self.target - self.epsilon)

            # Upper bound constraint: h(x) <= target + epsilon
            def upper_constraint(x):
                return (self.target + self.epsilon) - constraint_fun(x)

            # Constraint definition
            constraints = [
                {'type': 'ineq', 'fun': lower_constraint},  # h(x) >= target - epsilon
                {'type': 'ineq', 'fun': upper_constraint}   # h(x) <= target + epsilon
            ]
            x0 = np.array([prepared_input[idx] for idx in self.priorities_state.numerical_priorities.keys()])
            # Convert bounds dictionary to list of tuples in the same order as x0
            bounds_list = [bounds[idx] for idx in self.priorities_state.numerical_priorities.keys()]
            logger.info(f"Initial numerical values for optimization: {x0}")
            print('constraints: ', constraints)
            print('bounds: ', bounds_list)
            logger.info("Starting optimization...")
            result = minimize(
                objective_fun, 
                x0,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraints,
            )
            
            # The optimal solution is in result.x
            complete_solution = prepared_input.copy()
            for j, idx in enumerate(self.sample_state.shapley_values['numerical'].keys()):
                complete_solution[idx] = result.x[j]
            counterfactuals.append(complete_solution)
            # Print results
            logger.debug("Combination values: %s", values)
            logger.debug("Optimal numerical solution: %s", result.x)
            logger.debug("Complete solution (prepared_input with optimal numerical values): %s", self.model_pred([complete_solution]))
            logger.debug("Constraint value at solution: %s", constraint_fun(result.x))
            logger.debug("Maximized objective value: %s", -result.fun)
            logger.debug("-----")

        # TODO: create a loop to check if found counterfactuals meet the criteria (within epsilon of target) and use found counterfactuals as new samples

        ####################
        # 6. Select the best counterfactual based on total weight
        # if multiple found (only for cases with categorical features)
        logger.info("MINLP search completed. Selecting best counterfactual among {} candidates.".format(len(counterfactuals)))
        if len(counterfactuals) > 1:
            best_counterfactual = None
            best_weight = float('inf')
            for cf in counterfactuals:
                weight = self.calculate_total_weight(cf)
                if weight < best_weight:
                    best_weight = weight
                    best_counterfactual = cf
            counterfactual = best_counterfactual
        else:
            counterfactual = counterfactuals[0]
        return counterfactual

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
                return self.calculate_total_weight(temp_input)
            
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

    def display_priorities(self):
        """
        Display the priority functions and values for both numerical and categorical features.
        Similar to investigate_probability_distribution but shows the raw priority functions.
        """
        # Apply styling
        apply_style()

        # Display numerical feature priorities
        for idx, constraint in self.priorities['numerical'].items():

            if isinstance(constraint, dict) and 'function' in constraint:
                min_val = constraint['min']
                max_val = constraint['max']
                f = constraint['function']
                
                # Check if this is a non-actionable feature (function=None, min=max)
                if f is None or min_val == max_val:
                    print(f'Feature {idx} is non-actionable with fixed value: {min_val}')
                    continue
                
                # Calculate priority function values
                x_vals = np.linspace(min_val, max_val, 1000)
                priority_values = np.array([float(f(x)) for x in x_vals])

                # Create enhanced plot with gradient effects
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot priority function with enhanced styling
                ax.plot(x_vals, priority_values, label='Priority Function', 
                       color=get_line_color('theoretical'), linewidth=4, alpha=0.9,
                       solid_capstyle='round')
                
                # Add subtle fill under priority curve for depth
                ax.fill_between(x_vals, priority_values, alpha=0.2, 
                               color=get_line_color('theoretical'))
                
                # Mark the current sample value
                sample_value = self.sample[idx]
                if min_val <= sample_value <= max_val:
                    sample_priority = float(f(sample_value))
                    ax.plot(sample_value, sample_priority, 'o', 
                           color=get_line_color('empirical'), markersize=12, 
                           label=f'Current Sample Value ({sample_value:.3f})',
                           markeredgecolor=COLORS['dirty_white'], markeredgewidth=2)
                
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Priority Weight')
                ax.set_title(f'Priority Function for Numerical Feature {idx}')
                
                # Enhanced legend positioned to the right side of title, above plot
                legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                                 facecolor=COLORS['dark_background'], edgecolor=COLORS['dirty_white'],
                                 fontsize=16, loc='center left', bbox_to_anchor=(1.02, 0.9), ncol=1)
                legend.get_frame().set_alpha(0.9)
                # Make legend text dirty white
                for text in legend.get_texts():
                    text.set_color(COLORS['dirty_white'])
                
                # Apply numerical plot styling
                style_numerical_plot(ax)
                
                # Adjust layout to accommodate legend to the right
                plt.tight_layout()
                plt.subplots_adjust(right=0.75)  # Make room for legend on the right
                plt.show()
            else:
                # Old format fallback
                print(f'Feature {idx} has unexpected format: {constraint}')

        # Display categorical feature priorities
        for group_indices, possible_values in self.priorities['categorical'].items():
            
            # Extract categories and their weights
            categories = list(possible_values.keys())
            weights = np.array(list(possible_values.values()), dtype=float)
            
            # Create labels for the categories (convert tuples to strings for display)
            category_labels = [str(cat) if isinstance(cat, tuple) else str(cat) for cat in categories]
            
            # Calculate appropriate bar width based on number of categories
            num_categories = len(category_labels)
            if num_categories == 1:
                bar_width = 0.1  # Very narrow for single category
            elif num_categories == 2:
                bar_width = 0.4  # Narrow for 2 categories
            elif num_categories <= 4:
                bar_width = 0.5  # Narrow for 3-4 categories
            elif num_categories <= 6:
                bar_width = 0.6  # Moderate for 5-6 categories
            else:
                bar_width = 0.8  # Standard width for many categories
            
            # Create bar plot with enhanced styling and gradients
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create gradient colors for bars
            bar_colors = [get_bar_color(i) for i in range(len(category_labels))]
            
            bars = ax.bar(range(len(category_labels)), weights, width=bar_width, 
                         color=bar_colors, edgecolor=COLORS['dirty_white'], linewidth=2.0)
            
            # Apply gradient alpha effect to each bar
            for i, bar in enumerate(bars):
                # Create depth with alternating alpha and subtle gradients
                alpha_val = 0.7 + 0.2 * (i % 2)
                bar.set_alpha(alpha_val)
                
                # Add subtle inner border for depth with dark theme
                height = bar.get_height()
                if height > 0:
                    ax.add_patch(plt.Rectangle((bar.get_x() + 0.02, 0.01), 
                                             bar.get_width() - 0.04, height - 0.02,
                                             fill=False, edgecolor=COLORS['steel_gray'], 
                                             linewidth=0.8, alpha=0.6))
            
            # Highlight current sample values if they exist in the categories
            current_sample_combo = tuple(self.sample[idx] for idx in group_indices)
            if current_sample_combo in categories:
                current_idx = categories.index(current_sample_combo)
                bars[current_idx].set_edgecolor(get_line_color('empirical'))
                bars[current_idx].set_linewidth(4)
                # Add marker on top of the current sample bar
                ax.plot(current_idx, weights[current_idx] + max(weights) * 0.05, 'v', 
                       color=get_line_color('empirical'), markersize=15,
                       markeredgecolor=COLORS['dirty_white'], markeredgewidth=2)
            
            # For single category, adjust x-axis limits to center the bar better
            if num_categories == 1:
                ax.set_xlim(-1, 1)
            
            # Add value labels with enhanced dark theme styling
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                label_text = f'{weight:.3f}'
                if i == categories.index(current_sample_combo) if current_sample_combo in categories else -1:
                    label_text += ' (Current)'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weights) * 0.01, 
                       label_text, ha='center', va='bottom', fontsize=16,
                       fontweight='bold', color=COLORS['dirty_white'],
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['dark_background'], 
                                alpha=0.8, edgecolor=COLORS['dirty_white'], linewidth=1.5))
            
            ax.set_xlabel('Category Combinations')
            ax.set_ylabel('Priority Weight')
            ax.set_title(f'Priority Weights for Categorical Features {group_indices}')
            ax.set_xticks(range(len(category_labels)))
            ax.set_xticklabels(category_labels, rotation=45, ha='right')
            
            # Apply categorical plot styling
            style_categorical_plot(ax, num_categories)
            
            plt.tight_layout()
            plt.show()


    # def _calculate_target_for_combination(self, combination_id, combo, cat_combinations, 
    #                                     shap_dict, priorities_for_search, basic_prediction):
    #     """Helper to calculate target value for a specific categorical combination."""
    #     try:
    #         cat_shap = shap_dict['categorical']
    #         num_shap = shap_dict['numerical']
            
    #         # Calculate categorical contribution
    #         result = 0.0
    #         if combo:  # If there are categorical features
    #             max_idx = max(combo.keys()) if combo else 0
    #             dummy_x = np.zeros(max_idx + 1, dtype=float)
                
    #             for idx, val in combo.items():
    #                 dummy_x[idx] = val
                
    #             for feature_indices in priorities_for_search['categorical'].keys():
    #                 shap_value = cat_shap[feature_indices]
    #                 current_values = tuple(float(dummy_x[idx]) for idx in feature_indices)
    #                 sample_values = tuple(float(self.sample[idx]) for idx in feature_indices)
    #                 target_exemplar_values = tuple(float(self.target_exemplar[idx]) for idx in feature_indices)
                    
    #                 if current_values == sample_values:
    #                     result += 0.0
    #                 elif current_values == target_exemplar_values:
    #                     result += shap_value
    #                 else:
    #                     return None  # Invalid combination
            
    #         # Calculate coefficient times original
    #         coef_times_original = 0.0
    #         for key in priorities_for_search['numerical'].keys():
    #             if key in num_shap:
    #                 # Use pre-calculated coefficient from shap_coeffs instead of recalculating
    #                 temp_unit_phi = self.sample_state.shap_coeffs[key]
    #                 coef_times_original += temp_unit_phi * self.sample[key]
            
    #         # Return adjusted target
    #         return self.target - result - basic_prediction + coef_times_original
            
    #     except Exception as e:
    #         logger.warning(f"Error calculating target for combination {combination_id}: {e}")
    #         return None

