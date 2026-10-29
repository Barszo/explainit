"""Priority scores of candidate counterfactuals.

The *priority benefit* (higher is better) is the quantity the MINLP search
maximises: the sum of the numerical priority functions plus the categorical
weights.
"""

import numpy as np

from explainit.explainers.minlp.intervals import in_allowed_intervals
from explainit.logging_config import logger


def total_weight(values, numerical_priorities, categorical_priorities):
    """Priority benefit of ``values`` (a full feature vector).

    Immutable features (``function=None``) contribute nothing.

    Raises:
        ValueError: If a numerical value is outside its range or in a
            zero-priority gap, or a categorical combination is not allowed.
    """
    total = 0

    for feature_id, feature_config in numerical_priorities.items():
        if feature_id >= len(values):
            raise ValueError(f"Not enough values provided. Missing value for index {feature_id}")

        current_value = values[feature_id]

        if isinstance(feature_config, dict):
            function = feature_config.get("function")
            if function is not None:
                min_val = feature_config["min"]
                max_val = feature_config["max"]

                if current_value < min_val or current_value > max_val:
                    raise ValueError(
                        f"Value {current_value} at index {feature_id} is outside allowed range [{min_val}, {max_val}]"
                    )
                allowed_intervals = feature_config.get("allowed_intervals")
                if allowed_intervals and not in_allowed_intervals(
                    float(current_value), allowed_intervals
                ):
                    raise ValueError(
                        f"Value {current_value} at index {feature_id} is in a zero-priority gap; "
                        f"allowed intervals: {allowed_intervals}"
                    )

                total += function(current_value)

    for feature_indices, categorical_weights in categorical_priorities.items():
        group_values = []
        for idx in feature_indices:
            if idx >= len(values):
                raise ValueError(f"Not enough values provided. Missing value for index {idx}")
            group_values.append(float(values[idx]))

        value_tuple = tuple(group_values)

        if value_tuple not in categorical_weights:
            allowed_combinations = list(categorical_weights.keys())
            raise ValueError(
                f"Invalid categorical combination: {value_tuple} for features {feature_indices}. "
                f"Allowed combinations: {allowed_combinations}"
            )

        total += categorical_weights[value_tuple]

    return total


def priority_benefit(values, numerical_priorities, categorical_priorities):
    """:func:`total_weight` as a float, or ``None`` when ``values`` is not allowed.

    The optimiser can still produce out-of-range values, so callers get
    ``None`` instead of an exception.
    """
    try:
        return float(total_weight(values, numerical_priorities, categorical_priorities))
    except ValueError as exc:
        logger.debug("Priority benefit unavailable for candidate: %s", exc)
        return None


def feature_priority(config, value: float) -> float:
    """Priority of one numerical feature at ``value`` (0 if immutable or invalid)."""
    fn = config.get("function") if isinstance(config, dict) else None
    if fn is None:
        return 0.0
    try:
        w = float(np.asarray(fn(float(value))).squeeze())
    except Exception:
        return 0.0
    return w if np.isfinite(w) else 0.0


def priority_peaks(numerical_priorities, grid_size: int = 1000) -> dict[int, tuple]:
    """``{feature_index: (peak_value, peak_priority)}`` for mutable features.

    The peak is the argmax of the priority function over the feature's allowed
    intervals, found on a grid (no model calls).
    """
    peaks: dict[int, tuple] = {}
    for idx, cfg in numerical_priorities.items():
        if not isinstance(cfg, dict) or cfg.get("function") is None:
            continue
        intervals = cfg.get("allowed_intervals") or [(cfg["min"], cfg["max"])]
        best_v, best_w = None, float("-inf")
        for lo, hi in intervals:
            xs = np.linspace(float(lo), float(hi), int(grid_size))
            ws = np.array([feature_priority(cfg, v) for v in xs])
            k = int(ws.argmax())
            if ws[k] > best_w:
                best_w, best_v = float(ws[k]), float(xs[k])
        if best_v is not None:
            peaks[idx] = (best_v, best_w)
    return peaks
