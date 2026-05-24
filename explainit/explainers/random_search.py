
from explainit.logging_config import logger
from explainit.utils.priority_plots import (
    plot_priorities as _plot_priorities,
    plot_probability_distributions as _plot_probability_distributions,
    sample_numeric_value as _sample_numeric_value,
)

import numpy as np
import random


class RandomSearchExplainer:
    def __init__(self, model_pred, priorities, sample, target,):
        self.model_pred = model_pred
        self.priorities = priorities
        self.sample = sample
        self.target = target
        self.filtered_priorities = self.filter_priorities()

    def filter_priorities(self):
        """
        Filters the priorities dictionary by removing unactionable features.
        Unactionable numerical features (with priority None or 0) are replaced with the sample value.
        """
        
        new_priorities = {'numerical': {}, 'categorical': {}}

        # Numerical: replace None or 0 with sample value
        for idx, val in self.priorities['numerical'].items():
            if val is None or val == 0:
                new_priorities['numerical'][idx] = self.sample[idx]
            else:
                new_priorities['numerical'][idx] = val

        # Categorical: remove elements with value 0
        for group, mapping in self.priorities['categorical'].items():
            filtered_mapping = {k: v for k, v in mapping.items() if v != 0}
            if filtered_mapping:
                new_priorities['categorical'][group] = filtered_mapping

        return new_priorities

    def sample_numeric_value(self, constraint, max_tries=100):
        """Monte-Carlo sampler (rejection + grid fallback). Thin wrapper around
        :func:`explainit.utils.priority_plots.sample_numeric_value` so the
        sampling logic stays in lock-step with the distribution plots.
        """
        return _sample_numeric_value(constraint, max_tries=max_tries)

    def display_priorities(self, exemplar=None, *, save_dir="images", show=False, feature_names=None):
        """Plot the configured priorities (numerical + categorical).

        Delegates to ``explainit.utils.priority_plots.plot_priorities``; pass
        ``save_dir=None`` to skip writing files and ``show=True`` to open the
        plots interactively.
        """
        return _plot_priorities(
            self.priorities,
            sample=self.sample,
            exemplar=exemplar,
            feature_names=feature_names,
            save_dir=save_dir,
            show=show,
        )

    def investigate_probability_distribution(self, n_samples=10000, *, save_dir="images",
                                             show=False, feature_names=None):
        """Render theoretical vs empirical probability distributions for the
        priorities (numerical) and normalised category probabilities
        (categorical).

        Delegates to
        :func:`explainit.utils.priority_plots.plot_probability_distributions`,
        passing this explainer's own ``sample_numeric_value`` so the empirical
        histogram comes from the exact sampler the search loop uses.
        """
        return _plot_probability_distributions(
            self.priorities,
            n_samples=n_samples,
            sampler=self.sample_numeric_value,
            feature_names=feature_names,
            save_dir=save_dir,
            show=show,
        )

    def calculate_preference_score(self, sample):
        """
        Calculate overall preference score for a sample.
        
        Args:
            sample: List of feature values
        
        Returns:
            preference_score: Float value representing overall preference
        """
        scores = []
        
        # Calculate scores for numerical features
        for idx, constraint in self.priorities['numerical'].items():
            if isinstance(constraint, dict) and 'function' in constraint:
                # Get weight from preference function
                weight = float(np.asarray(constraint['function'](sample[idx])).squeeze())
                scores.append(weight)
        
        # Calculate scores for categorical features
        for group_indices, possible_values in self.priorities['categorical'].items():
            # Get current sample combination
            current_combo = tuple(sample[idx] for idx in group_indices)
            # Get weight for this combination (default to 0 if not found)
            weight = possible_values.get(current_combo, 0)
            scores.append(float(weight))
        
        # Overall score is the product of all weights (geometric mean approach)
        # This ensures that low preference in any feature significantly reduces overall score
        if len(scores) == 0:
            return 1.0
        return float(np.sum(scores))
    
    def get_preference_breakdown(self, sample):
        """
        Get detailed breakdown of preference scores for each feature.
        
        Args:
            sample: List of feature values
        
        Returns:
            breakdown: Dictionary with 'numerical' and 'categorical' contributions
        """
        breakdown = {
            'numerical': {},
            'categorical': {},
            'overall': 0.0
        }
        
        all_scores = []
        
        # Calculate scores for numerical features
        for idx, constraint in self.priorities['numerical'].items():
            if isinstance(constraint, dict) and 'function' in constraint:
                # Get weight from preference function
                weight = float(np.asarray(constraint['function'](sample[idx])).squeeze())
                breakdown['numerical'][idx] = {
                    'value': sample[idx],
                    'weight': weight,
                    'actionable': True
                }
                all_scores.append(weight)
            elif constraint is None or (isinstance(constraint, (int, float)) and constraint == 0):
                # Unactionable feature - fixed at sample value, doesn't affect preference score
                breakdown['numerical'][idx] = {
                    'value': sample[idx],
                    'weight': None,
                    'actionable': False
                }
                # Don't add to all_scores - unactionable features don't contribute to preference
        
        # Calculate scores for categorical features
        for group_indices, possible_values in self.priorities['categorical'].items():
            # Get current sample combination
            current_combo = tuple(sample[idx] for idx in group_indices)
            # Get weight for this combination (default to 0 if not found)
            weight = possible_values.get(current_combo, 0)
            breakdown['categorical'][group_indices] = {
                'combination': current_combo,
                'weight': float(weight)
            }
            all_scores.append(float(weight))
        
        # Calculate overall score
        breakdown['overall'] = float(np.sum(all_scores)) if len(all_scores) > 0 else 0.0
        
        return breakdown

    def generate_random_samples(self, expected_counterfactuals=5, max_iterations=10000, epsilon=0.05, random_seed=None, use_monte_carlo=True, max_tries=100, return_top_n=None):
        """
        Generate random samples based on filtered priorities.
        Only keep samples whose prediction is within epsilon of the target.
        
        Args:
            expected_counterfactuals: desired number of counterfactuals to find
            max_iterations: maximum number of iterations to try
            epsilon: acceptable deviation from target prediction
            random_seed: seed for reproducibility
            use_monte_carlo: if True, use Monte Carlo sampling for numerical features; otherwise use uniform sampling
            max_tries: maximum number of tries for rejection sampling in numerical features (only if use_monte_carlo is True)
            return_top_n: if specified, return only the top N most preferable samples (default: return all)
        
        Returns:
            list of samples and their corresponding predictions
        """
        # Only set seed if random_seed is provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        n_features = len(self.sample)
        best_samples = []
        best_predictions = []
        best_scores = []
        best_iteration_found = []
        n_candidates_per_cf = getattr(self, 'n_candidates_per_cf', 1)
        for cf_idx in range(expected_counterfactuals):
            candidates = []
            candidate_preds = []
            candidate_scores = []
            candidate_iters = []
            found = 0
            for i in range(max_iterations):
                sample = np.zeros(n_features)
                for idx, constraint in self.filtered_priorities['numerical'].items():
                    if isinstance(constraint, dict) and 'function' in constraint:
                        sample[idx] = self.sample_numeric_value(constraint, max_tries) if use_monte_carlo else np.random.uniform(constraint['min'], constraint['max'])
                    else:
                        sample[idx] = constraint
                for group_indices, possible_values in self.filtered_priorities['categorical'].items():
                    combos = list(possible_values.keys())
                    weights = np.array(list(possible_values.values()), dtype=float)
                    allowed_mask = weights > 0
                    if not np.any(allowed_mask):
                        raise ValueError(f"All combinations for categorical group {group_indices} are forbidden (weight = 0). At least one combination must have weight > 0.")
                    if use_monte_carlo:
                        allowed_combos = [combos[j] for j in range(len(combos)) if allowed_mask[j]]
                        allowed_weights = weights[allowed_mask]
                        allowed_weights = allowed_weights / allowed_weights.sum()
                        sel = allowed_combos[np.random.choice(len(allowed_combos), p=allowed_weights)]
                    else:
                        allowed_combos = [combos[j] for j in range(len(combos)) if allowed_mask[j]]
                        sel = random.choice(allowed_combos)
                    for j, idx in enumerate(group_indices):
                        sample[idx] = sel[j]
                try:
                    pred = self.model_pred(sample.reshape(1, -1))[0]
                    if abs(pred - self.target) <= epsilon:
                        candidates.append(sample.copy())
                        candidate_preds.append(pred)
                        candidate_scores.append(self.calculate_preference_score(sample))
                        candidate_iters.append(i + 1)
                        found += 1
                        if found >= n_candidates_per_cf:
                            break
                except Exception as e:
                    logger.warning(f"Could not get prediction for sample: {e}")
            if candidates:
                logger.info(f"CF {cf_idx+1}: Found {len(candidates)} valid candidates (requested: {n_candidates_per_cf})")
                # Log top 5 candidates by preference score
                top_n = min(5, len(candidate_scores))
                sorted_indices = np.argsort(candidate_scores)[::-1]
                logger.info(f"CF {cf_idx+1}: Top {top_n} candidates by preference score:")
                for rank in range(top_n):
                    idx = sorted_indices[rank]
                    logger.info(f"  Rank {rank+1}: Score={candidate_scores[idx]:.4f}, Iter={candidate_iters[idx]}, Pred={candidate_preds[idx]:.4f}, Sample={candidates[idx]}")
                best_idx = sorted_indices[0]
                logger.info(f"CF {cf_idx+1}: Selected candidate Rank 1 with Score={candidate_scores[best_idx]:.4f}, Iter={candidate_iters[best_idx]}, Pred={candidate_preds[best_idx]:.4f}, Sample={candidates[best_idx]}")
                best_samples.append(candidates[best_idx])
                best_predictions.append(candidate_preds[best_idx])
                best_scores.append(candidate_scores[best_idx])
                best_iteration_found.append(candidate_iters[best_idx])
            else:
                logger.info(f"No valid candidates found for CF {cf_idx+1}")
        return best_samples, best_predictions, best_scores, best_iteration_found

    def generate_for_binary(self, expected_counterfactuals=100, max_iterations=10000, target_class=1, threshold=0.5, random_seed=None, use_monte_carlo=True, max_tries=100, return_top_n=None, n_candidates_per_cf=1):
        """
        Generate random samples for binary classification.
        Only keep samples whose prediction crosses the threshold in the desired direction.
        
        Args:
            expected_counterfactuals: desired number of counterfactuals to find
            max_iterations: maximum number of iterations to try
            target_class: desired class (0 or 1)
            threshold: decision threshold for binary classification
            random_seed: seed for reproducibility
            use_monte_carlo: if True, use Monte Carlo sampling for numerical features; otherwise use uniform sampling
            max_tries: maximum number of tries for rejection sampling in numerical features (only if use_monte_carlo is True)
            return_top_n: if specified, return only the top N most preferable samples (default: return all)
        
        Returns:
            list of samples and their corresponding predictions
        """
        # Only set seed if random_seed is provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        samples = []
        predictions = []
        preference_scores = []
        iteration_found = []  # Track iteration number when each CF was found
        
        n_features = len(self.sample)
        best_samples = []
        best_predictions = []
        best_scores = []
        best_iteration_found = []
        # Use n_candidates_per_cf argument directly
        for cf_idx in range(expected_counterfactuals):
            candidates = []
            candidate_preds = []
            candidate_scores = []
            candidate_iters = []
            found = 0
            for i in range(max_iterations):
                sample = np.zeros(n_features)
                for idx, constraint in self.filtered_priorities['numerical'].items():
                    if isinstance(constraint, dict) and 'function' in constraint:
                        sample[idx] = self.sample_numeric_value(constraint, max_tries) if use_monte_carlo else np.random.uniform(constraint['min'], constraint['max'])
                    else:
                        sample[idx] = constraint
                for group_indices, possible_values in self.filtered_priorities['categorical'].items():
                    combos = list(possible_values.keys())
                    weights = np.array(list(possible_values.values()), dtype=float)
                    allowed_mask = weights > 0
                    if not np.any(allowed_mask):
                        raise ValueError(f"All combinations for categorical group {group_indices} are forbidden (weight = 0). At least one combination must have weight > 0.")
                    if use_monte_carlo:
                        allowed_combos = [combos[j] for j in range(len(combos)) if allowed_mask[j]]
                        allowed_weights = weights[allowed_mask]
                        allowed_weights = allowed_weights / allowed_weights.sum()
                        sel = allowed_combos[np.random.choice(len(allowed_combos), p=allowed_weights)]
                    else:
                        allowed_combos = [combos[j] for j in range(len(combos)) if allowed_mask[j]]
                        sel = random.choice(allowed_combos)
                    for j, idx in enumerate(group_indices):
                        sample[idx] = sel[j]
                try:
                    pred = self.model_pred(sample.reshape(1, -1))[0]
                    if target_class == 1:
                        is_valid = pred > threshold
                    else:
                        is_valid = pred < threshold
                    if is_valid:
                        candidates.append(sample.copy())
                        candidate_preds.append(pred)
                        candidate_scores.append(self.calculate_preference_score(sample))
                        candidate_iters.append(i + 1)
                        found += 1
                        if found >= n_candidates_per_cf:
                            break
                except Exception as e:
                    logger.warning(f"Could not get prediction for sample: {e}")
            if candidates:
                logger.info(f"CF {cf_idx+1}: Found {len(candidates)} valid candidates (requested: {n_candidates_per_cf})")
                # Log top 5 candidates by preference score
                top_n = min(5, len(candidate_scores))
                sorted_indices = np.argsort(candidate_scores)[::-1]
                logger.info(f"CF {cf_idx+1}: Top {top_n} candidates by preference score:")
                for rank in range(top_n):
                    idx = sorted_indices[rank]
                    logger.info(f"  Rank {rank+1}: Score={candidate_scores[idx]:.4f}, Iter={candidate_iters[idx]}, Pred={candidate_preds[idx]:.4f}, Sample={candidates[idx]}")
                best_idx = sorted_indices[0]
                logger.info(f"CF {cf_idx+1}: Selected candidate Rank 1 with Score={candidate_scores[best_idx]:.4f}, Iter={candidate_iters[best_idx]}, Pred={candidate_preds[best_idx]:.4f}, Sample={candidates[best_idx]}")
                best_samples.append(candidates[best_idx])
                best_predictions.append(candidate_preds[best_idx])
                best_scores.append(candidate_scores[best_idx])
                best_iteration_found.append(candidate_iters[best_idx])
            else:
                logger.info(f"No valid candidates found for CF {cf_idx+1}")
        return best_samples, best_predictions, best_scores, best_iteration_found
