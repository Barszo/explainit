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
           - For each numerical feature i: `unit_phi_i = shapley_i / (closest_sample_i - sample_i)`
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
                       match either the sample or closest_sample values
                       
        Example:
            >>> cat_combos = {0: {3: 1.0, 4: 0.0}, 1: {3: 0.0, 4: 1.0}}
            >>> shap_dict = {'numerical': {0: 0.1, 1: -0.2}, 'categorical': {(3,4): 0.15}}
            >>> targets, coeffs = explainer.extract_for_linear_search(cat_combos, shap_dict, 
            ...                                                      priorities, 0.5)
            >>> print(targets)  # {0: 0.23, 1: 0.08}  # Different targets per combination
            >>> print(coeffs)   # {0: {'coeff': 2.5, 'min_max': (0, 1)}, ...}
        """
        print('cat_combinations:', cat_combinations)
        print('shap_dict:', shap_dict)
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
        else:
            # no categorical features
            new_targets[0] = self.target - basic_prediction + coef_times_original

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
            logger.info(f'Processing combination_id: {combination_id} with target: {temp_target}')
            temp_combo = cat_combinations[combination_id]
            logger.info(f'temp_combo: {temp_combo}')
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
        if len(combo_and_initial_solutions) == 0:
            # Try to find intelligent bounds adjustments
            bounds_adjustment = self.find_required_bounds_adjustments(
                cat_combinations, shap_dict, priorities_for_search, 
                basic_prediction, target_for_combo, shap_coefficients
            )
            
            if bounds_adjustment['feasible_with_shifts']:
                logger.warning("No feasible solutions found with current bounds, but solutions possible with adjusted bounds:")
                for feature_idx, adjustment in bounds_adjustment['required_shifts'].items():
                    logger.warning(f"Feature {feature_idx}: {adjustment['justification']}")
                    logger.warning(f"  Current bounds: {adjustment['original_bounds']}")
                    logger.warning(f"  Suggested bounds: {adjustment['suggested_bounds']}")
                
                raise ValueError(
                    f"No feasible initial solutions found for any categorical combination. "
                    f"However, solutions may be possible with adjusted bounds. "
                    f"Consider adjusting bounds for features: {list(bounds_adjustment['required_shifts'].keys())}"
                )
            else:
                raise ValueError("No feasible initial solutions found for any categorical combination, even with bounds adjustments.")
                
        return combo_and_initial_solutions, priorities_for_search

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
                        dummy_x, shap_dict, self.sample, self.closest_sample, 
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
    
    def _calculate_target_for_combination(self, combination_id, combo, cat_combinations, 
                                        shap_dict, priorities_for_search, basic_prediction):
        """Helper to calculate target value for a specific categorical combination."""
        try:
            cat_shap = shap_dict['categorical']
            num_shap = shap_dict['numerical']
            
            # Calculate categorical contribution
            result = 0.0
            if combo:  # If there are categorical features
                max_idx = max(combo.keys()) if combo else 0
                dummy_x = np.zeros(max_idx + 1, dtype=float)
                
                for idx, val in combo.items():
                    dummy_x[idx] = val
                
                for feature_indices in priorities_for_search['categorical'].keys():
                    shap_value = cat_shap[feature_indices]
                    current_values = tuple(float(dummy_x[idx]) for idx in feature_indices)
                    sample_values = tuple(float(self.sample[idx]) for idx in feature_indices)
                    closest_values = tuple(float(self.closest_sample[idx]) for idx in feature_indices)
                    
                    if current_values == sample_values:
                        result += 0.0
                    elif current_values == closest_values:
                        result += shap_value
                    else:
                        return None  # Invalid combination
            
            # Calculate coefficient times original
            coef_times_original = 0.0
            for key in priorities_for_search['numerical'].keys():
                if key in num_shap:
                    temp_unit_phi = num_shap[key] / (self.closest_sample[key] - self.sample[key])
                    coef_times_original += temp_unit_phi * self.sample[key]
            
            # Return adjusted target
            return self.target - result - basic_prediction + coef_times_original
            
        except Exception as e:
            logger.warning(f"Error calculating target for combination {combination_id}: {e}")
            return None

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
            logger.debug("Prepared input: %s", prepared_input)

            def constraint_wrapper(x, shap_dict, sample, closest_sample, priorities_for_search, basic_prediction, ready_input):
                for idx in shap_dict['numerical'].keys():
                    ready_input[idx] = x[idx]

                return self.constraint_function(ready_input, shap_dict, sample, closest_sample, priorities_for_search, basic_prediction)

            def objective_wrapper(x, priorities_for_search, ready_input):
                for idx in self.shap_dict['numerical'].keys():
                    ready_input[idx] = x[idx]
                return self.calculate_total_weight(ready_input)

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
            logger.debug("Combination values: %s", values)
            logger.debug("Optimal numerical solution: %s", result.x)
            logger.debug("Complete solution (prepared_input with optimal numerical values): %s", self.model_pred([complete_solution]))
            logger.debug("Constraint value at solution: %s", constraint_fun(result.x))
            logger.debug("Maximized objective value: %s", -result.fun)
            logger.debug("-----")

        # TODO: create a loop to check if found counterfactuals meet the criteria (within epsilon of target) and use found counterfactuals as new samples
        return counterfactuals
    

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
                print(f'Feature {idx} is unactionable with fixed value: {constraint}')

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
