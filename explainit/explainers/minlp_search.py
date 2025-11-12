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

class MINLSearchExplainer:
    def __init__(self, model_pred, priorities, sample, target, dataset, closest_sample_epsilon=0.01, epsilon=0.01):
        self.model_pred = model_pred
        self.priorities = priorities
        self.sample = sample
        self.sample_for_search = sample.copy()
        self.target = target
        # self.filtered_priorities = self.filter_priorities()
        self.dataset = dataset
        self.epsilon = epsilon

        cls_sampl = self.find_closest_elem(epsilon=closest_sample_epsilon)
        assert len(cls_sampl) == 1, "There should be exactly one closest sample"
        self.closest_sample = cls_sampl[0]

    # def filter_priorities(self):
    #     """
    #     Filters the priorities dictionary by removing unactionable features.
    #     Unactionable numerical features (with priority 0) are replaced with the sample value.
    #     """
        
    #     new_priorities = {'numerical': {}, 'categorical': {}}

    #     # Numerical: replace 0 with sample value
    #     for idx, val in self.priorities['numerical'].items():
    #         if val == 0:
    #             new_priorities['numerical'][idx] = self.sample[idx]
    #         else:
    #             new_priorities['numerical'][idx] = val

    #     # Categorical: remove elements with value 0
    #     for group, mapping in self.priorities['categorical'].items():
    #         filtered_mapping = {k: v for k, v in mapping.items() if v != 0}
    #         if filtered_mapping:
    #             new_priorities['categorical'][group] = filtered_mapping

    #     return new_priorities
    

    def get_rows_in_priorities(self):
        """
        Filters dataset rows based on priorities. Removes rows that do not meet the criteria defined in weights.
        Parameters:
        data_np: numpy array of shape (n_samples, n_features)
        weights: {'numerical': {idx: func or value}, 'categorical': {group: {vals: weight}}}
        """
        data_np = self.dataset.copy()

        num_w = self.priorities['numerical']
        cat_w = self.priorities['categorical']

        # Categorical features
        zero_cat_entries = [
            (group, cat_vals)
            for group, mapping in cat_w.items()
            for cat_vals, v in mapping.items()
            if v == 0
        ]

        def remv_cat(np_arr, idx_tup,  vals):
            mask = ~(np.all(data_np[:, idx_tup] == vals, axis=1))
            return np_arr[mask]

        for elem in zero_cat_entries:
            data_np = remv_cat(data_np, elem[0], elem[1])
            if data_np.size == 0:
                raise Exception("There are no elements fulfilling the requirements")

        # Numerical
        zero_num_entries = [(idx, val) for idx, val in num_w.items() if callable(val)]

        def check_ranges(np_arr, idx, f):
            mask = f(np_arr[:, idx]) != 0
            return np_arr[mask]

        for idx, f in zero_num_entries:
            data_np = check_ranges(data_np, idx, f)
            if data_np.size == 0:
                raise Exception("There are no elements fulfilling the requirements")

        return data_np

    # Find the elements from dataset that is the closest to the wanted 
    def find_closest_elem(self, epsilon: float) -> list:
        """
        Finds the elements in the dataset that are closest to the desired target value +- epsilon.
        Returns the actual samples of all such elements.
        """
        logger.info("Finding the closest element in the dataset based on the specified priorities...")
        filtered_data = self.get_rows_in_priorities()
        pred = self.model_pred(filtered_data)

        min_dist = np.abs(pred - self.target).min()
        if min_dist > epsilon:
            raise ValueError(f"No elements found within the specified epsilon of {epsilon}. Closest distance is {min_dist}.")

        idx_all = np.where(np.abs(pred - self.target) == min_dist)[0]

        return filtered_data[idx_all]

##################################
# Shapley values for MINLP search
##################################

    def calc_shapley(self, r: list, use_approximation: bool = False, num_samples: int = 200) -> list:
        """
        Calculate Shapley values for feature importance in counterfactual explanations.
        
        ## What are Shapley Values?
        
        Shapley values are a concept from cooperative game theory that fairly distributes 
        the "contribution" of each feature to a model's prediction. In our context, they 
        measure how much each feature contributes to the difference between the current 
        sample and the closest sample that achieves our target prediction.
        
        ## Mathematical Foundation
        
        For a feature i, the Shapley value φᵢ is calculated as:
        
        φᵢ = Σ_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] × [f(S∪{i}) - f(S)]
        
        Where:
        - N is the set of all features
        - S is a subset of features not including i
        - f(S) is the model prediction using features in S from closest_sample, others from reference
        - n is the total number of features
        - The weight |S|!(n-|S|-1)!/n! ensures fair attribution
        
        ## How It Works
        
        1. **Reference vs Target**: We compare the original sample (r) with the closest 
           sample that achieves our target prediction (self.closest_sample)
        
        2. **Marginal Contributions**: For each feature, we calculate its marginal 
           contribution across all possible subsets of other features
        
        3. **Weighted Average**: The final Shapley value is a weighted average of these 
           marginal contributions, where weights ensure mathematical fairness
        
        ## Efficiency Features
        
        This vectorized implementation:
        - Batches model predictions for significant speedup
        - Supports both exact calculation and sampling-based approximation
        - Handles categorical features by grouping related columns
        - Reduces computational complexity from O(2ⁿ × n) individual predictions to batched operations
        
        Args:
            r: Reference sample (original input)
            use_approximation: If True, uses Monte Carlo sampling for faster computation
            num_samples: Number of samples for approximation (only used if use_approximation=True)
            
        Returns:
            numpy.ndarray: Shapley values for each feature/feature group
            
        Example:
            >>> explainer = MINLSearchExplainer(model, priorities, sample, target, dataset)
            >>> shapley_vals = explainer.calc_shapley(sample)
            >>> print(f"Feature contributions: {shapley_vals}")
        """
        cat_groups = [list(elem) for elem in self.priorities['categorical'].keys()]

        def flatten_args_vectorized(func):
            """Vectorized version of flatten_args that handles batch inputs"""
            def wrapper(x_batch):
                if x_batch.ndim == 1:
                    x_batch = x_batch.reshape(1, -1)
                
                flattened_batch = []
                for x in x_batch:
                    flattened = []
                    for elem in x:
                        if isinstance(elem, tuple):
                            flattened.extend(elem)
                        else:
                            flattened.append(elem)
                    flattened_batch.append(flattened)
                
                return func(np.array(flattened_batch))
            return wrapper

        def z_of_S_batch(S_list, r, x):
            """Create batch of hybrid vectors for list of subsets S"""
            z_batch = []
            for S in S_list:
                z = r.copy()
                for i in S:
                    z[i] = x[i]
                z_batch.append(z)
            return np.array(z_batch)

        def combine_categories(original, cat_groups, cat_idx):
            """Helper function for combining categorical features into tuples"""
            new_list = []
            idx_original = list(range(len(original)))
            passed_groups = []
            for idx in idx_original:
                if idx not in cat_idx:
                    new_list.append(original[idx])
                else:
                    if idx in passed_groups:
                        continue
                    # Find the group containing this index
                    for elem in cat_groups:
                        if idx in elem:
                            current_idx = elem
                            break
                    # Extract categorical values as tuple
                    cat_vals = [original[i] for i in current_idx]
                    new_list.append(tuple(cat_vals))
                    passed_groups.extend(current_idx)
            return new_list

        def shapley_value_vectorized(i, x, r, f, n):
            """Vectorized exact Shapley value calculation"""
            others = [j for j in range(n) if j != i]
            
            # Collect all subsets and their weights
            all_subsets = []
            all_subsets_with_i = []
            all_weights = []
            
            total_subsets = 2 ** len(others)
            print(f"Computing vectorized Shapley for feature {i}, processing {total_subsets} subsets in batches...")
            
            for k in range(len(others) + 1):
                for S in itertools.combinations(others, k):
                    S = set(S)
                    weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    
                    all_subsets.append(S)
                    all_subsets_with_i.append(S | {i})
                    all_weights.append(weight)
            
            # Create batches for prediction
            z_without_batch = z_of_S_batch(all_subsets, r, x)
            z_with_batch = z_of_S_batch(all_subsets_with_i, r, x)
            
            # Batch predictions
            pred_without_batch = f(z_without_batch)
            pred_with_batch = f(z_with_batch)
            
            # Ensure we get proper scalar values
            if pred_without_batch.ndim > 1:
                pred_without_batch = pred_without_batch.flatten()
            if pred_with_batch.ndim > 1:
                pred_with_batch = pred_with_batch.flatten()
            
            # Calculate weighted differences
            diff_batch = pred_with_batch - pred_without_batch
            weighted_contributions = np.array(all_weights) * diff_batch
            
            total = np.sum(weighted_contributions)
            return total

        def shapley_value_approximate_vectorized(i, x, r, f, n, num_samples=200):
            """Vectorized approximate Shapley values using sampling"""
            others = [j for j in range(n) if j != i]
            
            print(f"Computing vectorized approximate Shapley for feature {i}, using {num_samples} samples...")
            
            # Generate all random subsets at once
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
            z_without_batch = z_of_S_batch(all_subsets, r, x)
            z_with_batch = z_of_S_batch(all_subsets_with_i, r, x)
            
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

        # Process categorical combinations if needed
        x = self.closest_sample
        f = self.model_pred

        if cat_groups:
            # Assert consecutive integers for categorical groups
            for elem in cat_groups:
                assert all(elem[j] == elem[j-1] + 1 for j in range(1, len(elem))), \
                    f"Elements in group {elem} are not consecutive integers."

            cat_idx = [item for sublist in cat_groups for item in sublist]
            x = combine_categories(self.closest_sample, cat_groups, cat_idx)
            r = combine_categories(r, cat_groups, cat_idx)
            f = flatten_args_vectorized(self.model_pred)

        n = len(r)
        
        # Choose between exact and approximate calculation
        if use_approximation:
            phi = [shapley_value_approximate_vectorized(i, x, r, f, n, num_samples) for i in range(n)]
        else:
            phi = [shapley_value_vectorized(i, x, r, f, n) for i in range(n)]

        return np.array(phi, dtype=float)
    
########################################
# Prepare for search of counterfactuals
########################################

    def create_modified_priorities(self):
        """
        Create modified priorities dictionary based on SHAP values and sample data.
        
        This method processes the calculated Shapley values to determine which categorical 
        combinations should be considered during the counterfactual search. The logic is:
        
        For categorical features:
        - If SHAP value is 0: Feature doesn't contribute to prediction difference, 
          so keep only the category from the original sample
        - If SHAP value is not 0: Feature contributes to prediction difference, 
          so allow categories from both sample and closest_sample
        
        This reduces the search space by eliminating categorical combinations that 
        don't contribute to achieving the target prediction.
        
        Returns:
            tuple: (modified_priorities_dict, all_combinations_dict, shap_values_dict)
                - modified_priorities_dict: Updated priorities with filtered categorical options
                - all_combinations_dict: All possible categorical combinations to explore
                - shap_values_dict: SHAP values organized by feature type and index
        """


        # Create a deep copy of the original priorities
        modified_priorities = copy.deepcopy(self.priorities)
        
        # Dictionary to store SHAP values by index
        shap_values_dict = {
            'numerical': {},
            'categorical': {}
        }
        
        # Store allowed combinations for each categorical group
        allowed_combinations_by_group = []
        all_categorical_indices = []
        
        # Process numerical features first
        for feature_id in self.priorities['numerical'].keys():
            if feature_id < len(self.shap_vals):
                shap_values_dict['numerical'][feature_id] = self.shap_vals[feature_id]
            else:
                shap_values_dict['numerical'][feature_id] = 0.0
        
        # Create mapping from feature groups to SHAP indices
        shap_index = len(self.priorities['numerical'])
        
        # Process categorical features
        for feature_indices, categorical_weights in self.priorities['categorical'].items():
            # Extract values for this categorical group from sample and closest_sample
            sample_values = tuple(float(self.sample[idx]) for idx in feature_indices)
            closest_values = tuple(float(self.closest_sample[idx]) for idx in feature_indices)
            
            # Get SHAP value for this categorical group
            if shap_index < len(self.shap_vals):
                shap_val = self.shap_vals[shap_index]
            else:
                shap_val = 0.0
            
            # Store SHAP value for this categorical group
            shap_values_dict['categorical'][feature_indices] = shap_val
            
            # Create new categorical weights dictionary and collect allowed combinations
            new_categorical_weights = {}
            group_combinations = []
            
            if shap_val == 0:
                # Keep only the category from sample
                if sample_values in categorical_weights:
                    new_categorical_weights[sample_values] = categorical_weights[sample_values]
                    group_combinations.append(sample_values)
            else:
                # Keep categories from both sample and closest_sample
                if sample_values in categorical_weights:
                    new_categorical_weights[sample_values] = categorical_weights[sample_values]
                    group_combinations.append(sample_values)
                if closest_values in categorical_weights and closest_values != sample_values:
                    new_categorical_weights[closest_values] = categorical_weights[closest_values]
                    group_combinations.append(closest_values)
            
            # Update the modified priorities
            modified_priorities['categorical'][feature_indices] = new_categorical_weights
            
            # Store for cross-product calculation
            allowed_combinations_by_group.append(group_combinations)
            all_categorical_indices.extend(feature_indices)
            
            shap_index += 1
        
        # Generate all possible combinations across all categorical groups
        all_combinations = {}
        combination_id = 0
        
        # Use itertools.product to get all combinations across groups
        for combination_tuple in itertools.product(*allowed_combinations_by_group):
            # Create dictionary mapping each categorical index to its value
            combination_dict = {}
            
            # Track position in the flattened combination
            group_index = 0
            
            for feature_indices in self.priorities['categorical'].keys():
                group_values = combination_tuple[group_index]
                for i, idx in enumerate(feature_indices):
                    combination_dict[idx] = group_values[i]
                group_index += 1
            
            all_combinations[combination_id] = combination_dict
            combination_id += 1
        
        return modified_priorities, all_combinations, shap_values_dict



    def constraint_function(self, x, shap_dict, sample, closest_sample, priorities_for_search, basic_prediction):
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
            closest_sample: Closest sample feature values array/list
            priorities_for_search: Modified priorities dictionary
            basic_prediction: Model prediction for the original sample
            
        Returns:
            float: Estimated prediction value based on Shapley value approximation
        """
        num_shap=shap_dict['numerical']
        cat_shap=shap_dict['categorical']

        result = basic_prediction

        for i, key in enumerate(priorities_for_search['numerical'].keys()):

            temp_unit_phi = num_shap[key]/(closest_sample[key] - sample[key])
            
            result += temp_unit_phi * (x[i] - sample[key])
        
        for i, key in enumerate(priorities_for_search['categorical'].keys()):
            feature_indices = key
            shap_value = cat_shap[feature_indices]
            
            # Extract current values from x
            current_values = tuple(float(x[idx]) for idx in feature_indices) #tuple(float(x[len(priorities_for_search['numerical']) + i + j]) for j in range(len(feature_indices)))
            
            sample_values = tuple(float(sample[idx]) for idx in feature_indices)
            closest_values = tuple(float(closest_sample[idx]) for idx in feature_indices)

            if current_values == sample_values:
                result += 0.0
            elif current_values == closest_values:
                result += shap_value
            else:
                raise ValueError("Invalid categorical combination encountered for indices {}: {}".format(feature_indices, current_values))

        return result
    
    def extract_for_linear_search(self, cat_combinations, shap_dict, priorities_for_search, basic_prediction):

        num_shap = shap_dict['numerical']
        cat_shap = shap_dict['categorical']

        new_targets = {}
        coef_times_original = 0.0
        shap_coeffs = {}

        for i, key in enumerate(priorities_for_search['numerical'].keys()):
            temp_unit_phi = num_shap[key]/(self.closest_sample[key] - self.sample[key])
            if type(priorities_for_search['numerical'][key]) is dict:
                min_val = priorities_for_search['numerical'][key]['min']
                max_val = priorities_for_search['numerical'][key]['max']
                shap_coeffs[key] = {'coeff':temp_unit_phi, 'min_max':(min_val, max_val)}
                coef_times_original += temp_unit_phi * self.sample[key]

        
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


            for i, key in enumerate(priorities_for_search['categorical'].keys()):
                feature_indices = key
                shap_value = cat_shap[feature_indices]
            
                # Extract current values from x
                current_values = tuple(float(dummy_x[idx]) for idx in feature_indices) #tuple(float(x[len(priorities_for_search['numerical']) + i + j]) for j in range(len(feature_indices)))
            
                sample_values = tuple(float(self.sample[idx]) for idx in feature_indices)
                closest_values = tuple(float(self.closest_sample[idx]) for idx in feature_indices)
            
                if current_values == sample_values:
                    result += 0.0
                elif current_values == closest_values:
                    result += shap_value
                else:
                    raise ValueError("Invalid categorical combination encountered for indices {}: {}".format(feature_indices, current_values))

            new_targets[combo_idx] = self.target - result - basic_prediction + coef_times_original #- unactionable_sum

        return new_targets, shap_coeffs
    



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

    def confirm_existence_of_solution_for_combo(self, use_approximation=False, num_samples=200):

        # 1. Calculate SHAP values for the sample
        self.shap_vals = self.calc_shapley(self.sample, use_approximation=use_approximation, num_samples=num_samples)
        logger.info(f'SHAP values: {self.shap_vals}')
        # 2. Prepare variables for next steps, including categorical combinations
        priorities_for_search, cat_combinations, shap_dict = self.create_modified_priorities()
        self.shap_dict = shap_dict
        # 3. Extract targets for linear search for each categorical combination
        basic_prediction=self.model_pred([self.sample])[0]

        target_for_combo, shap_coefficients = self.extract_for_linear_search(cat_combinations, shap_dict, priorities_for_search, basic_prediction=basic_prediction)
        indices_to_modify = list(shap_coefficients.keys())
        coeff_to_linear_search = [elem['coeff'] for elem in shap_coefficients.values()]
        bounds_for_linear_search = [elem['min_max'] for elem in shap_coefficients.values()]

        # 4. For each categorical combination, solve the linear programming problem to find at least one solution
        combo_and_initial_solutions = {}
        for combination_id, temp_target in target_for_combo.items():
            temp_combo = cat_combinations[combination_id]
            dummy_x = self.sample.copy()

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
                logger.info(f'sample: {self.sample}')
                logger.info(f'dummy_x: {dummy_x}')
                assert abs(self.constraint_function(dummy_x, shap_dict, self.sample, self.closest_sample, priorities_for_search, basic_prediction=self.model_pred([self.sample])[0])-self.target) <= self.epsilon, \
                    f"Constraint function value {self.constraint_function(dummy_x, shap_dict, self.sample, self.closest_sample, priorities_for_search, basic_prediction=self.model_pred([self.sample])[0])} exceeds tolerance {self.epsilon}"
                logger.info(f'constraint_function: {self.constraint_function(dummy_x, shap_dict, self.sample, self.closest_sample, priorities_for_search, basic_prediction=self.model_pred([self.sample])[0])}')

        return combo_and_initial_solutions, priorities_for_search
    

    def calculate_total_weight(self, values):
        """
        Calculate total weight (cost) based on priorities dictionary and input values list.
        
        This function computes the total cost of changing features from the original sample
        to the proposed counterfactual values. It validates that all values are within 
        allowed ranges for numerical features and acceptable categorical combinations.
        
        The weight calculation follows the priority functions defined in self.priorities:
        - For numerical features: applies the weight function to the feature value
        - For categorical features: uses predefined weights for specific combinations
        - For unactionable features (priority 0): contributes 0 to total weight
        
        Args:
            values: List of values where each element corresponds to its feature index
        
        Returns:
            float: Total calculated weight (cost) for the proposed feature changes
        
        Raises:
            ValueError: If any value is outside allowed range or invalid categorical combination
        """
        total_weight = 0
        
        # Process numerical features
        for feature_id, feature_config in self.priorities['numerical'].items():
            if feature_id >= len(values):
                raise ValueError(f"Not enough values provided. Missing value for index {feature_id}")
            
            current_value = values[feature_id]
            
            if feature_config == 0:  # unactionable
                # No validation needed, contributes 0 to total weight
                pass
            else:
                # Validate value is within min/max range
                min_val = feature_config['min']
                max_val = feature_config['max']
                
                if current_value < min_val or current_value > max_val:
                    raise ValueError(f"Value {current_value} at index {feature_id} is outside allowed range [{min_val}, {max_val}]")
                
                # Apply the weight function
                function = feature_config['function']
                weight = function(current_value)
                total_weight += weight
        
        # Process categorical features
        for feature_indices, categorical_weights in self.priorities['categorical'].items():
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
        def create_bounds(priorities, sample):
            bounds = []
            num_part = priorities['numerical']
            for idx, value in num_part.items():
                if type(value) is dict:
                    temp_bound = (value['min'], value['max'])
                elif value == 0:
                    temp_bound = (sample[idx], sample[idx])
                bounds.append(temp_bound)

            return bounds
        bounds = create_bounds(self.priorities, self.sample)
        logger.info(f"Bounds for numerical features: {bounds}")
        # x0 = np.array([self.closest_sample[idx] for idx in self.priorities['numerical'].keys()])
        # epsilon = 0.01
        init_vals_per_combo, priorities_for_search = self.confirm_existence_of_solution_for_combo(use_approximation=shap_approx, num_samples=num_samples)
        logger.info(f"Initial values per combination: {init_vals_per_combo}")
        logger.info(f"Priorities for search: {priorities_for_search}")

        counterfactuals = []
        for i, values in init_vals_per_combo.items():
            prepared_input = self.sample.copy()
            initial_numerical = values['initial_solution']
            cat_combo = values['categorical_combo']
            for idx, val in cat_combo.items():
                prepared_input[idx] = val
            for idx, val in initial_numerical.items():
                prepared_input[idx] = val
            logger.debug("Prepared input:", prepared_input)

            def constraint_wrapper(x, shap_dict, sample, closest_sample, priorities_for_search, basic_prediction, ready_input):
                for idx in shap_dict['numerical'].keys():
                    ready_input[idx] = x[idx]

                return self.constraint_function(ready_input, shap_dict, sample, closest_sample, priorities_for_search, basic_prediction)

            # constraint_fun = lambda x: constraint_wrapper(x, shap_dict, sample, explainer.closest_sample, priorities_for_search, basic_prediction=model_pred([sample])[0], ready_input=prepared_input)
            # print(constraint_fun(x))
            def objective_wrapper(x, priorities_for_search, ready_input):
                for idx in self.shap_dict['numerical'].keys():
                    ready_input[idx] = x[idx]
                return self.calculate_total_weight(ready_input)

            # objective_fun = lambda x: -objective_wrapper(x, priorities_for_search, prepared_input)

            constraint_fun = lambda x: constraint_wrapper(x, self.shap_dict, self.sample, self.closest_sample, priorities_for_search, basic_prediction=self.model_pred([self.sample])[0], ready_input=prepared_input.copy())

            objective_fun = lambda x: -objective_wrapper(x, priorities_for_search, prepared_input.copy())

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
            x0 = np.array([prepared_input[idx] for idx in self.priorities['numerical'].keys()])
            # Solve
            result = minimize(
                objective_fun, 
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )
            
            # The optimal solution is in result.x
            complete_solution = prepared_input.copy()
            for j, idx in enumerate(self.shap_dict['numerical'].keys()):
                complete_solution[idx] = result.x[j]
            counterfactuals.append(complete_solution)
            # Print results
            logger.debug("Combination values: ", values)
            logger.debug("Optimal numerical solution:", result.x)
            logger.debug("Complete solution (prepared_input with optimal numerical values):", self.model_pred([complete_solution]))
            logger.debug("Constraint value at solution:", constraint_fun(result.x))
            logger.debug("Maximized objective value:", -result.fun)
            logger.debug("-----")

        # TODO: create a loop to check if found counterfactuals meet the criteria (within epsilon of target) and use found counterfactuals as new samples
        return counterfactuals