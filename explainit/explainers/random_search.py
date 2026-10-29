"""Random (Monte Carlo) counterfactual search.

Candidates are sampled from the region the priorities allow, with more
desirable values sampled more often; the model checks each candidate, and the
most preferable valid ones are returned.
"""

import numpy as np

from explainit._random import make_rngs
from explainit.logging_config import logger
from explainit.sampling import sample_category, sample_numeric_value as _sample_numeric_value
from explainit.utils.priority_plots import (
    plot_priorities as _plot_priorities,
    plot_probability_distributions as _plot_probability_distributions,
)


class RandomSearchExplainer:
    """Counterfactual search by sampling from the priorities.

    Args:
        model_pred: Callable mapping a 2-D array to one prediction per row.
        priorities: ``{"numerical": ..., "categorical": ...}``; see the README.
        sample: The point to explain.
        target: The desired prediction (regression-style search).
    """

    def __init__(self, model_pred, priorities, sample, target):
        self.model_pred = model_pred
        self.priorities = priorities
        self.sample = sample
        self.target = target
        self.filtered_priorities = self.filter_priorities()
        self._np_rng, self._py_rng = make_rngs(None)

    def filter_priorities(self):
        """
        Filters the priorities dictionary by removing unactionable features.
        Unactionable numerical features (with priority None or 0) are replaced with the sample value.
        """

        new_priorities = {"numerical": {}, "categorical": {}}

        # Numerical: replace None or 0 with sample value
        for idx, val in self.priorities["numerical"].items():
            if val is None or (isinstance(val, (int, float)) and val == 0):
                new_priorities["numerical"][idx] = self.sample[idx]
            elif isinstance(val, dict) and val.get("function") is None:
                # Non-actionable feature: fixed at its configured value
                # (falls back to the sample value).
                new_priorities["numerical"][idx] = val.get("min", self.sample[idx])
            else:
                new_priorities["numerical"][idx] = val

        # Categorical: remove elements with value 0
        for group, mapping in self.priorities["categorical"].items():
            filtered_mapping = {k: v for k, v in mapping.items() if v != 0}
            if filtered_mapping:
                new_priorities["categorical"][group] = filtered_mapping

        return new_priorities

    def display_priorities(self, exemplar=None, *, save_dir=None, show=True, feature_names=None):
        """Plot the configured priorities (numerical + categorical).

        Delegates to ``explainit.utils.priority_plots.plot_priorities``. Pass
        ``save_dir`` to also write PNG files, and ``show=False`` to skip
        opening the plots (useful in scripts and tests).
        """
        return _plot_priorities(
            self.priorities,
            sample=self.sample,
            exemplar=exemplar,
            feature_names=feature_names,
            save_dir=save_dir,
            show=show,
        )

    def investigate_probability_distribution(
        self, n_samples=10000, *, save_dir=None, show=True, feature_names=None
    ):
        """Render theoretical vs empirical probability distributions for the
        priorities (numerical) and normalised category probabilities
        (categorical).

        Delegates to
        :func:`explainit.utils.priority_plots.plot_probability_distributions`,
        passing this explainer's own ``sample_numeric_value`` so the empirical
        histogram comes from the exact sampler the search loop uses. Pass
        ``save_dir`` to also write PNG files, and ``show=False`` to skip
        opening the plots.
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
        for idx, constraint in self.priorities["numerical"].items():
            if isinstance(constraint, dict) and constraint.get("function") is not None:
                # Get weight from preference function
                weight = float(np.asarray(constraint["function"](sample[idx])).squeeze())
                scores.append(weight)

        # Calculate scores for categorical features
        for group_indices, possible_values in self.priorities["categorical"].items():
            # Get current sample combination
            current_combo = tuple(sample[idx] for idx in group_indices)
            # Get weight for this combination (default to 0 if not found)
            weight = possible_values.get(current_combo, 0)
            scores.append(float(weight))

        # Overall score is the sum of the per-feature priorities, so each
        # actionable feature contributes a value in [0, 1].
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
        breakdown = {"numerical": {}, "categorical": {}, "overall": 0.0}

        all_scores = []

        # Calculate scores for numerical features
        for idx, constraint in self.priorities["numerical"].items():
            if isinstance(constraint, dict) and constraint.get("function") is not None:
                # Get weight from preference function
                weight = float(np.asarray(constraint["function"](sample[idx])).squeeze())
                breakdown["numerical"][idx] = {
                    "value": sample[idx],
                    "weight": weight,
                    "actionable": True,
                }
                all_scores.append(weight)
            elif (
                constraint is None
                or (isinstance(constraint, (int, float)) and constraint == 0)
                or (isinstance(constraint, dict) and constraint.get("function") is None)
            ):
                # Unactionable feature - fixed at sample value, doesn't affect preference score
                breakdown["numerical"][idx] = {
                    "value": sample[idx],
                    "weight": None,
                    "actionable": False,
                }
                # Don't add to all_scores - unactionable features don't contribute to preference

        # Calculate scores for categorical features
        for group_indices, possible_values in self.priorities["categorical"].items():
            # Get current sample combination
            current_combo = tuple(sample[idx] for idx in group_indices)
            # Get weight for this combination (default to 0 if not found)
            raw_weight = possible_values.get(current_combo, 0)
            weight = 0.0 if raw_weight is None else float(raw_weight)
            # A categorical feature is actionable only if the search may switch
            # it to more than one combination, i.e. at least two combinations
            # carry a selectable (non-None) weight. A feature pinned to the
            # sample's category (all other combinations set to None) is not
            # actionable.
            selectable = sum(1 for w in possible_values.values() if w is not None)
            actionable = selectable > 1
            breakdown["categorical"][group_indices] = {
                "combination": current_combo,
                "weight": weight,
                "actionable": actionable,
            }
            all_scores.append(weight)

        # Calculate overall score
        breakdown["overall"] = float(np.sum(all_scores)) if len(all_scores) > 0 else 0.0

        return breakdown

    def sample_numeric_value(self, constraint, max_tries=100):
        """Draw one value for a numerical feature; see :func:`explainit.sampling.sample_numeric_value`."""
        return _sample_numeric_value(constraint, max_tries=max_tries, rng=self._np_rng)

    def generate_random_samples(
        self,
        expected_counterfactuals=5,
        max_iterations=10000,
        epsilon=0.05,
        random_seed=None,
        use_monte_carlo=True,
        max_tries=100,
        return_top_n=None,
        n_candidates_per_cf=1,
    ):
        """Find counterfactuals whose prediction is within ``epsilon`` of the target.

        Args:
            expected_counterfactuals: Number of counterfactuals to search for.
            max_iterations: Candidates sampled at most per counterfactual.
            epsilon: Accepted distance between the prediction and the target.
            random_seed: Seed for reproducible results.
            use_monte_carlo: Sample in proportion to the priorities; otherwise
                uniformly within the allowed ranges and categories.
            max_tries: Rejection-sampling attempts per numerical value.
            return_top_n: Return only the ``n`` most preferable results,
                ordered by decreasing preference score.
            n_candidates_per_cf: Valid candidates to collect per
                counterfactual; the most preferable one is kept.

        Returns:
            tuple: ``(counterfactuals, predictions, scores, iterations)``, four
            lists with one entry per counterfactual found: the feature vector,
            the model prediction, the preference score (see
            :meth:`calculate_preference_score`) and the iteration at which it
            was found. Fewer than ``expected_counterfactuals`` entries are
            returned when the search budget runs out.
        """
        return self._search(
            lambda pred: abs(pred - self.target) <= epsilon,
            expected_counterfactuals=expected_counterfactuals,
            max_iterations=max_iterations,
            random_seed=random_seed,
            use_monte_carlo=use_monte_carlo,
            max_tries=max_tries,
            return_top_n=return_top_n,
            n_candidates_per_cf=n_candidates_per_cf,
        )

    def generate_for_binary(
        self,
        expected_counterfactuals=100,
        max_iterations=10000,
        target_class=1,
        threshold=0.5,
        random_seed=None,
        use_monte_carlo=True,
        max_tries=100,
        return_top_n=None,
        n_candidates_per_cf=1,
    ):
        """Find counterfactuals classified as ``target_class`` by a binary classifier.

        A candidate is valid when its predicted probability is above
        ``threshold`` (``target_class=1``) or below it (``target_class=0``).
        The other arguments and the return value are as in
        :meth:`generate_random_samples`.
        """
        return self._search(
            lambda pred: pred > threshold if target_class == 1 else pred < threshold,
            expected_counterfactuals=expected_counterfactuals,
            max_iterations=max_iterations,
            random_seed=random_seed,
            use_monte_carlo=use_monte_carlo,
            max_tries=max_tries,
            return_top_n=return_top_n,
            n_candidates_per_cf=n_candidates_per_cf,
        )

    def _sample_candidate(self, use_monte_carlo, max_tries):
        """One candidate inside the allowed region (known issue RS-1: starts from zeros)."""
        candidate = np.zeros(len(self.sample))
        for idx, constraint in self.filtered_priorities["numerical"].items():
            if isinstance(constraint, dict) and "function" in constraint:
                candidate[idx] = (
                    self.sample_numeric_value(constraint, max_tries)
                    if use_monte_carlo
                    else self._np_rng.uniform(constraint["min"], constraint["max"])
                )
            else:
                candidate[idx] = constraint
        for group, weights in self.filtered_priorities["categorical"].items():
            combination = sample_category(
                group, weights, use_monte_carlo, self._np_rng, self._py_rng
            )
            for j, idx in enumerate(group):
                candidate[idx] = combination[j]
        return candidate

    def _search(
        self,
        is_valid,
        *,
        expected_counterfactuals,
        max_iterations,
        random_seed,
        use_monte_carlo,
        max_tries,
        return_top_n,
        n_candidates_per_cf,
    ):
        """Shared loop of the public search methods; ``is_valid(prediction)`` decides validity."""
        self._np_rng, self._py_rng = make_rngs(random_seed)

        best_samples, best_predictions, best_scores, best_iterations = [], [], [], []
        for cf_idx in range(expected_counterfactuals):
            candidates, predictions, scores, iterations = [], [], [], []
            for i in range(max_iterations):
                candidate = self._sample_candidate(use_monte_carlo, max_tries)
                try:
                    pred = self.model_pred(candidate.reshape(1, -1))[0]
                    if is_valid(pred):
                        candidates.append(candidate.copy())
                        predictions.append(pred)
                        scores.append(self.calculate_preference_score(candidate))
                        iterations.append(i + 1)
                        if len(candidates) >= n_candidates_per_cf:
                            break
                except Exception as e:  # known issue RS-2
                    logger.warning(f"Could not get prediction for sample: {e}")

            if not candidates:
                logger.info(f"No valid candidates found for CF {cf_idx + 1}")
                continue
            best = self._log_and_pick_best(
                cf_idx, candidates, predictions, scores, iterations, n_candidates_per_cf
            )
            best_samples.append(candidates[best])
            best_predictions.append(predictions[best])
            best_scores.append(scores[best])
            best_iterations.append(iterations[best])

        return self._top_n(
            (best_samples, best_predictions, best_scores, best_iterations), return_top_n
        )

    @staticmethod
    def _log_and_pick_best(cf_idx, candidates, predictions, scores, iterations, requested):
        """Index of the most preferable candidate; logs the top five."""
        logger.info(
            f"CF {cf_idx + 1}: Found {len(candidates)} valid candidates (requested: {requested})"
        )
        top_n = min(5, len(scores))
        order = np.argsort(scores)[::-1]
        logger.info(f"CF {cf_idx + 1}: Top {top_n} candidates by preference score:")
        for rank in range(top_n):
            idx = order[rank]
            logger.info(
                f"  Rank {rank + 1}: Score={scores[idx]:.4f}, Iter={iterations[idx]}, Pred={predictions[idx]:.4f}, Sample={candidates[idx]}"
            )
        best = order[0]
        logger.info(
            f"CF {cf_idx + 1}: Selected candidate Rank 1 with Score={scores[best]:.4f}, Iter={iterations[best]}, Pred={predictions[best]:.4f}, Sample={candidates[best]}"
        )
        return best

    @staticmethod
    def _top_n(results, n):
        """Keep the ``n`` results with the highest preference score."""
        if n is None:
            return results
        samples, predictions, scores, iterations = results
        order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[: int(n)]
        return (
            [samples[k] for k in order],
            [predictions[k] for k in order],
            [scores[k] for k in order],
            [iterations[k] for k in order],
        )
