"""Random (Monte Carlo) counterfactual search.

Candidates are sampled from the region the priorities allow, with more
desirable values sampled more often; the model checks each candidate, and the
most preferable valid ones are returned.
"""

import numpy as np

from explainit._random import make_rngs
from explainit.explanation import Explanation
from explainit.logging_config import logger
from explainit.sampling import sample_category, sample_numeric_value
from explainit.targets import resolve_target
from explainit.utils.priority_plots import (
    plot_priorities as _plot_priorities,
    plot_probability_distributions as _plot_probability_distributions,
)


class RandomSearchExplainer:
    """Counterfactual search by sampling from the priorities.

    Args:
        model: Callable mapping a 2-D array to one prediction per row, e.g.
            ``lambda X: classifier.predict_proba(X)[:, 1]``.
        priorities: ``{"numerical": ..., "categorical": ...}``; see the README.
        max_iterations: Candidates sampled at most per counterfactual.
        candidates_per_counterfactual: Valid candidates collected per
            counterfactual; the most preferable one is kept.
        use_monte_carlo: Sample in proportion to the priorities; otherwise
            uniformly within the allowed ranges and categories.
        max_tries: Rejection-sampling attempts per numerical value.
    """

    def __init__(
        self,
        model,
        priorities,
        *,
        max_iterations=10000,
        candidates_per_counterfactual=1,
        use_monte_carlo=True,
        max_tries=100,
    ):
        self.model = model
        self.priorities = priorities
        self.max_iterations = max_iterations
        self.candidates_per_counterfactual = candidates_per_counterfactual
        self.use_monte_carlo = use_monte_carlo
        self.max_tries = max_tries

    def explain(
        self,
        x,
        *,
        target=None,
        tolerance=0.05,
        target_class=None,
        threshold=0.5,
        n_counterfactuals=5,
        top_n=None,
        random_seed=None,
    ) -> Explanation:
        """Find counterfactuals for the sample ``x``.

        Give either ``target`` (a prediction value, reached within
        ``tolerance``) or ``target_class`` (1: probability above ``threshold``;
        0: below it).

        Args:
            x: The sample to explain (1-D).
            target: Desired prediction value.
            tolerance: Accepted distance between prediction and ``target``.
            target_class: Desired class of a binary classifier (0 or 1).
            threshold: Decision threshold for ``target_class``.
            n_counterfactuals: Number of counterfactuals to search for. Fewer
                are returned when the search budget runs out.
            top_n: Return only the ``top_n`` most preferable counterfactuals,
                ordered by decreasing preference score.
            random_seed: Seed for reproducible results.

        Returns:
            Explanation: ``diagnostics["iterations"]`` holds the iteration at
            which each counterfactual was found.
        """
        goal = resolve_target(target, tolerance, target_class, threshold)
        sample = np.array(x, dtype=float)
        filtered = filter_priorities(self.priorities, sample)
        np_rng, py_rng = make_rngs(random_seed)

        found = []
        for cf_idx in range(n_counterfactuals):
            candidates = []
            for i in range(self.max_iterations):
                candidate = sample_candidate(
                    filtered, len(sample), self.use_monte_carlo, self.max_tries, np_rng, py_rng
                )
                try:
                    pred = self.model(candidate.reshape(1, -1))[0]
                    if goal.is_reached(pred):
                        candidates.append(
                            (candidate.copy(), pred, self.preference_score(candidate), i + 1)
                        )
                        if len(candidates) >= self.candidates_per_counterfactual:
                            break
                except Exception as e:  # known issue RS-2
                    logger.warning(f"Could not get prediction for sample: {e}")

            if not candidates:
                logger.info(f"No valid candidates found for CF {cf_idx + 1}")
                continue
            found.append(_log_and_pick_best(cf_idx, candidates, self.candidates_per_counterfactual))

        if top_n is not None:
            found = sorted(found, key=lambda c: c[2], reverse=True)[: int(top_n)]
        return Explanation.from_lists(
            sample,
            [c[0] for c in found],
            [c[1] for c in found],
            [c[2] for c in found],
            [True] * len(found),
            method="random_search",
            diagnostics={"iterations": [c[3] for c in found], "target": goal},
        )

    def preference_score(self, x) -> float:
        """Sum of the priorities of ``x``'s feature values (higher is better)."""
        return preference_score(self.priorities, x)

    def preference_breakdown(self, x) -> dict:
        """Per-feature priorities of ``x``; see :func:`preference_breakdown`."""
        return preference_breakdown(self.priorities, x)

    def plot_priorities(
        self, x=None, *, exemplar=None, feature_names=None, save_dir=None, show=True
    ):
        """Plot the priorities, marking ``x`` and ``exemplar`` if given.

        Pass ``save_dir`` to also write PNG files, and ``show=False`` to skip
        opening the plots.
        """
        return _plot_priorities(
            self.priorities,
            sample=x,
            exemplar=exemplar,
            feature_names=feature_names,
            save_dir=save_dir,
            show=show,
        )

    def plot_sampling_distributions(
        self, n_samples=10000, *, random_seed=None, feature_names=None, save_dir=None, show=True
    ):
        """Plot how the search samples each feature: theoretical vs. empirical distribution."""
        np_rng = make_rngs(random_seed)[0]
        return _plot_probability_distributions(
            self.priorities,
            n_samples=n_samples,
            sampler=lambda constraint: sample_numeric_value(
                constraint, max_tries=self.max_tries, rng=np_rng
            ),
            feature_names=feature_names,
            save_dir=save_dir,
            show=show,
        )


def filter_priorities(priorities, sample):
    """Priorities prepared for sampling.

    Immutable numerical features (``None``, ``0``, or no ``function``) are
    replaced by the fixed value they keep; zero-weight categories are removed.
    """
    filtered = {"numerical": {}, "categorical": {}}

    for idx, val in priorities["numerical"].items():
        if val is None or (isinstance(val, (int, float)) and val == 0):
            filtered["numerical"][idx] = sample[idx]
        elif isinstance(val, dict) and val.get("function") is None:
            # Fixed at its configured value, falling back to the sample value.
            filtered["numerical"][idx] = val.get("min", sample[idx])
        else:
            filtered["numerical"][idx] = val

    for group, mapping in priorities["categorical"].items():
        allowed = {k: v for k, v in mapping.items() if v != 0}
        if allowed:
            filtered["categorical"][group] = allowed

    return filtered


def sample_candidate(filtered, n_features, use_monte_carlo, max_tries, np_rng, py_rng):
    """One candidate inside the allowed region (known issue RS-1: starts from zeros)."""
    candidate = np.zeros(n_features)
    for idx, constraint in filtered["numerical"].items():
        if isinstance(constraint, dict) and "function" in constraint:
            candidate[idx] = (
                sample_numeric_value(constraint, max_tries, rng=np_rng)
                if use_monte_carlo
                else np_rng.uniform(constraint["min"], constraint["max"])
            )
        else:
            candidate[idx] = constraint
    for group, weights in filtered["categorical"].items():
        combination = sample_category(group, weights, use_monte_carlo, np_rng, py_rng)
        for j, idx in enumerate(group):
            candidate[idx] = combination[j]
    return candidate


def preference_score(priorities, x) -> float:
    """Sum of the numerical priorities and categorical weights of ``x``."""
    scores = []
    for idx, constraint in priorities["numerical"].items():
        if isinstance(constraint, dict) and constraint.get("function") is not None:
            scores.append(float(np.asarray(constraint["function"](x[idx])).squeeze()))
    for group, weights in priorities["categorical"].items():
        combination = tuple(x[idx] for idx in group)
        scores.append(float(weights.get(combination, 0)))  # known issue RS-3
    if len(scores) == 0:
        return 1.0  # known issue RS-5
    return float(np.sum(scores))


def preference_breakdown(priorities, x) -> dict:
    """Per-feature contributions to the preference score of ``x``.

    Returns:
        dict: ``{"numerical": {index: {"value", "weight", "actionable"}},
        "categorical": {group: {"combination", "weight", "actionable"}},
        "overall": total}``. Immutable features have weight ``None``.
    """
    breakdown = {"numerical": {}, "categorical": {}, "overall": 0.0}
    all_scores = []

    for idx, constraint in priorities["numerical"].items():
        if isinstance(constraint, dict) and constraint.get("function") is not None:
            weight = float(np.asarray(constraint["function"](x[idx])).squeeze())
            breakdown["numerical"][idx] = {"value": x[idx], "weight": weight, "actionable": True}
            all_scores.append(weight)
        elif (
            constraint is None
            or (isinstance(constraint, (int, float)) and constraint == 0)
            or (isinstance(constraint, dict) and constraint.get("function") is None)
        ):
            breakdown["numerical"][idx] = {"value": x[idx], "weight": None, "actionable": False}

    for group, weights in priorities["categorical"].items():
        combination = tuple(x[idx] for idx in group)
        raw_weight = weights.get(combination, 0)
        weight = 0.0 if raw_weight is None else float(raw_weight)
        # Actionable only if the search may pick more than one combination.
        selectable = sum(1 for w in weights.values() if w is not None)
        breakdown["categorical"][group] = {
            "combination": combination,
            "weight": weight,
            "actionable": selectable > 1,
        }
        all_scores.append(weight)

    breakdown["overall"] = float(np.sum(all_scores)) if len(all_scores) > 0 else 0.0
    return breakdown


def _log_and_pick_best(cf_idx, candidates, requested):
    """The most preferable ``(candidate, prediction, score, iteration)``; logs the top five."""
    scores = [c[2] for c in candidates]
    logger.info(
        f"CF {cf_idx + 1}: Found {len(candidates)} valid candidates (requested: {requested})"
    )
    top = min(5, len(scores))
    order = np.argsort(scores)[::-1]
    logger.info(f"CF {cf_idx + 1}: Top {top} candidates by preference score:")
    for rank in range(top):
        cand, pred, score, iteration = candidates[order[rank]]
        logger.info(
            f"  Rank {rank + 1}: Score={score:.4f}, Iter={iteration}, Pred={pred:.4f}, Sample={cand}"
        )
    best = candidates[order[0]]
    logger.info(
        f"CF {cf_idx + 1}: Selected candidate Rank 1 with Score={best[2]:.4f}, Iter={best[3]}, Pred={best[1]:.4f}, Sample={best[0]}"
    )
    return best
