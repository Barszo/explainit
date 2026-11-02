"""Stage 4.5: find a point on a hyperplane inside box bounds.

Solves ``coefficients . x = target`` subject to ``bounds`` with a linear
program; the solution warm-starts the nonlinear optimisation.
"""

import itertools
import warnings

import numpy as np
from scipy.optimize import linprog


def solve_linear_constraint(coefficients, target, bounds, method="auto", tolerance=1e-8):
    """Find ``x`` with ``|coefficients . x - target| < tolerance`` inside ``bounds``.

    Args:
        coefficients: Coefficients ``[a_1, ..., a_n]``.
        target: Right-hand side of ``a_1 x_1 + ... + a_n x_n = target``.
        bounds: ``[(min_1, max_1), ..., (min_n, max_n)]``.
        method: ``"auto"`` (HiGHS, then interior point) or a ``linprog`` method.
        tolerance: Accepted constraint violation.

    Returns:
        dict: ``success``, ``solution`` (array or ``None``), ``message`` and
        ``method_used``; on success also ``constraint_value`` and ``error``.
    """
    coefficients = np.array(coefficients)
    n = len(coefficients)

    if len(bounds) != n:
        return {
            "success": False,
            "solution": None,
            "message": "Bounds length must match coefficients length",
            "method_used": None,
        }

    if np.allclose(coefficients, 0):
        if abs(target) < tolerance:
            # Any point in bounds is a solution
            solution = np.array([(b[0] + b[1]) / 2 for b in bounds])
            return {
                "success": True,
                "solution": solution,
                "message": "Trivial solution (all coefficients zero)",
                "method_used": "analytical",
            }
        return {
            "success": False,
            "solution": None,
            "message": "No solution (all coefficients zero, target non-zero)",
            "method_used": None,
        }

    methods_to_try = ["highs", "interior-point"] if method == "auto" else [method]

    for current_method in methods_to_try:
        try:
            # The equality is written as two inequalities with a zero objective,
            # so linprog only looks for a feasible point.
            A_ub = np.array([coefficients, -coefficients])
            b_ub = np.array([target, -target])
            c = np.zeros(n)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method=current_method,
                    options={"presolve": True},
                )

            if result.success:
                constraint_value = np.dot(coefficients, result.x)
                error = abs(constraint_value - target)

                if error < tolerance:
                    return {
                        "success": True,
                        "solution": result.x,
                        "message": f"Solution found with error {error:.2e}",
                        "method_used": current_method,
                        "constraint_value": constraint_value,
                        "error": error,
                    }

        except Exception:
            continue

    return _corner_analysis(coefficients, target, bounds, tolerance)


def _corner_analysis(coefficients, target, bounds, tolerance):
    """Fallback: the box corner closest to the hyperplane, if within tolerance."""
    corners = []
    for bound in bounds:
        corners.append([bound[0], bound[1]])

    corner_points = list(itertools.product(*corners))

    best_error = float("inf")
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
            "success": True,
            "solution": best_solution,
            "message": f"Corner solution found with error {best_error:.2e}",
            "method_used": "corner_analysis",
            "constraint_value": np.dot(coefficients, best_solution),
            "error": best_error,
        }
    return {
        "success": False,
        "solution": None,
        "message": f"No feasible solution found. Best error: {best_error:.2e}",
        "method_used": "corner_analysis",
    }
