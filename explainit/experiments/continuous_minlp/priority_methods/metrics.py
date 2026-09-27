"""Per-counterfactual metrics shared by the priority branch and the paper studies.

Kept separate from the runners so the publication harness
(``explainit/experiments/paper``) computes exactly the same numbers as
``priority_methods/runner.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from explainit.experiments.continuous_minlp.priority_methods.methods import (
    compute_priority_score,
)

CHANGE_TOL = 1e-6


def compute_cf_metrics(
    x: np.ndarray,
    cf: Optional[np.ndarray],
    target: float,
    epsilon: float,
    model_predict,
    priorities: Dict[str, Any],
    max_priority: Optional[float] = None,
) -> Dict[str, Any]:
    """Validity / proximity / sparsity / priority metrics for one counterfactual.

    ``priorities`` must be a pristine build (the MINLP search mutates the dict
    it is given), and ``max_priority`` the output of
    :func:`~...methods.max_attainable_priority` for that same build.
    """
    if cf is None:
        return {
            "cf_prediction": None, "validity": False, "abs_pred_error": None,
            "l1": None, "l2": None, "n_changed": None, "sparsity_fraction": None,
            "priority_score": None, "priority_score_normalised": None,
        }
    cf = np.asarray(cf, dtype=float)
    cf_pred = float(model_predict(cf.reshape(1, -1))[0])
    abs_err = abs(cf_pred - float(target))
    diff = np.abs(cf - x)
    n_changed = int(np.sum(diff > CHANGE_TOL))
    priority = compute_priority_score(priorities, cf)
    return {
        "cf_prediction": cf_pred,
        "validity": bool(abs_err <= epsilon),
        "abs_pred_error": abs_err,
        "l1": float(np.sum(diff)),
        "l2": float(np.linalg.norm(cf - x)),
        "n_changed": n_changed,
        "sparsity_fraction": float(n_changed / len(x)),
        "priority_score": priority,
        "priority_score_normalised": (
            float(priority / max_priority) if max_priority else None
        ),
    }


__all__ = ["CHANGE_TOL", "compute_cf_metrics"]
