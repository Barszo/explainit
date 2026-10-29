"""Stages 6 and 7: evaluate, repair, select and polish candidates.

* :func:`evaluate_candidate` scores a candidate with the real model.
* :func:`restore_to_band` pulls a candidate that just missed the target band
  back into it.
* :func:`select_best_candidate` picks the best candidate of a pass.
* :func:`peak_lock_in` moves features to their priority peaks while other
  features compensate, after the search.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from explainit.logging_config import logger


@dataclass(frozen=True)
class RefinementContext:
    """What the refinement steps need from the current search state.

    Attributes:
        model: The real model.
        target: Target prediction.
        epsilon: Half-width of the target band.
        coefficients: Per-unit surrogate coefficients ``{column: c}``.
        immutable: Columns that must not change.
        surrogate: ``h(x)`` for a full feature vector.
        priority_benefit: Benefit of a full feature vector, or ``None``.
        feature_priority: ``feature_priority(column, value)``.
        clip: ``clip(column, value)`` onto the allowed region.
        priority_peaks: ``priority_peaks()`` -> ``{column: (value, priority)}``.
        feature_label: Display name of a column.
    """

    model: Callable
    target: float
    epsilon: float
    coefficients: dict | None
    immutable: set
    surrogate: Callable
    priority_benefit: Callable
    feature_priority: Callable
    clip: Callable
    priority_peaks: Callable
    feature_label: Callable


def _prediction(model, values) -> float:
    return float(np.asarray(model([list(values)])).reshape(-1)[0])


def evaluate_candidate(ctx: RefinementContext, cf) -> dict:
    """Score ``cf`` against the real model and the surrogate.

    Returns:
        dict: ``model_pred``; ``h_x`` (surrogate, NaN if it cannot be
        evaluated); ``distance`` ``|model_pred - target|``; ``priority``
        (benefit or ``None``); ``feasible`` (``distance <= epsilon``).
    """
    model_pred = float(ctx.model([cf])[0])
    try:
        h_x = float(ctx.surrogate(cf))
    except Exception:
        h_x = float("nan")
    distance = abs(model_pred - float(ctx.target))
    return {
        "model_pred": model_pred,
        "h_x": h_x,
        "distance": distance,
        "priority": ctx.priority_benefit(cf),
        "feasible": distance <= float(ctx.epsilon),
    }


def select_best_candidate(ctx: RefinementContext, counterfactuals):
    """Stage 6: the candidate with the highest priority benefit.

    Raises:
        RuntimeError: If there are no candidates.
    """
    logger.info("--- STAGE 6/6: Pick best counterfactual ---")
    logger.info(
        "Among %d candidate(s), pick the one with the highest priority "
        "benefit (calculate_total_weight). With no categorical features "
        "there is exactly one candidate.",
        len(counterfactuals),
    )
    if not counterfactuals:
        raise RuntimeError("No counterfactual candidates produced for selection.")
    if len(counterfactuals) == 1:
        return counterfactuals[0]

    best_counterfactual = None
    best_benefit = float("-inf")
    for cf in counterfactuals:
        benefit = ctx.priority_benefit(cf)
        logger.debug("Candidate priority benefit=%s", benefit)
        if benefit is not None and benefit > best_benefit:
            best_benefit = benefit
            best_counterfactual = cf
    if best_counterfactual is None:
        logger.warning(
            "No candidate had an evaluable priority benefit; falling back to the first candidate."
        )
        return counterfactuals[0]
    logger.info("Selected candidate priority benefit=%.4f", best_benefit)
    return best_counterfactual


def restore_to_band(ctx: RefinementContext, cf, eval_info, max_steps: int = 3):
    """Pull a near-miss candidate back into the target band.

    Least-norm Shapley-Newton correction: ``dx = -r c / ||c||^2`` moves the
    surrogate by exactly ``-r`` (the measured residual) while changing the
    features as little as possible, so most of the candidate's priority
    survives. Costs at most ``max_steps`` model calls.

    Returns:
        tuple: ``(cf, eval_info)`` when the point ends up inside the band,
        otherwise ``(None, None)``.
    """
    coeffs = ctx.coefficients
    if not coeffs:
        return None, None
    indices = [i for i in coeffs if i not in ctx.immutable and abs(float(coeffs[i])) > 1e-12]
    if not indices:
        return None, None
    denom = sum(float(coeffs[i]) ** 2 for i in indices)
    if denom < 1e-18:
        return None, None

    cur = np.asarray(cf, dtype=float).copy()
    residual = float(eval_info["model_pred"]) - float(ctx.target)
    for step in range(int(max_steps)):
        for i in indices:
            cur[i] = ctx.clip(i, float(cur[i]) - residual * float(coeffs[i]) / denom)
        residual = _prediction(ctx.model, cur) - float(ctx.target)
        if abs(residual) <= float(ctx.epsilon):
            restored = evaluate_candidate(ctx, list(cur))
            logger.info(
                "[Restoration] Candidate pulled into the band in %d "
                "step(s): |f - target| %.4f -> %.4f | priority %s -> %s.",
                step + 1,
                eval_info["distance"],
                restored["distance"],
                eval_info["priority"],
                restored["priority"],
            )
            return list(cur), restored
    logger.debug(
        "[Restoration] Failed after %d step(s); |f - target|=%.4f.",
        int(max_steps),
        abs(residual),
    )
    return None, None


def peak_lock_in(ctx: RefinementContext, cf, max_sweeps: int = 3, max_compensators: int = 3):
    """Stage 7: greedy peak lock-in with Shapley compensation.

    For each mutable numerical feature ``i`` (largest possible gain first),
    propose ``x_i = peak_i`` and pay for the surrogate change
    ``dh = c_i (peak_i - x_i)`` by moving the feature ``j`` whose priority
    loses the least: ``x_j -= dh / c_j``. One model call verifies the swap; if
    it leaves the band, one restoration step on ``j`` is tried. Up to
    ``max_compensators`` features ``j`` are tried. A move is kept only when the
    point stays in the band *and* the priority benefit increases, so ``cf``
    can never get worse.

    Returns:
        tuple: ``(cf, info)`` where ``info`` reports the priority before and
        after, the accepted moves and the model calls spent.
    """
    info = {
        "applied": False,
        "sweeps_run": 0,
        "accepted_moves": 0,
        "model_calls": 0,
        "priority_before": None,
        "priority_after": None,
        "priority_gain": 0.0,
        "final_prediction": None,
    }
    coeffs_all = ctx.coefficients
    if cf is None or not coeffs_all:
        return cf, info

    peaks = ctx.priority_peaks()
    actionable = [
        i for i in peaks if i not in ctx.immutable and abs(float(coeffs_all.get(i, 0.0))) > 1e-12
    ]
    if not actionable:
        return cf, info

    cur = np.asarray(cf, dtype=float).copy()
    cur_priority = ctx.priority_benefit(list(cur))
    if cur_priority is None:
        return cf, info
    info["applied"] = True
    info["priority_before"] = cur_priority

    logger.info("--- STAGE 7/7: Peak lock-in with Shapley compensation ---")
    logger.info(
        "Sweeping %d actionable feature(s); start priority=%.4f. Each move "
        "locks a feature on its priority peak and pays for the linearised "
        "prediction change with the cheapest compensating feature.",
        len(actionable),
        cur_priority,
    )

    for sweep in range(int(max_sweeps)):
        info["sweeps_run"] = sweep + 1
        accepted_in_sweep = 0
        order = sorted(
            actionable,
            key=lambda i: -(peaks[i][1] - ctx.feature_priority(i, cur[i])),
        )
        for i in order:
            peak_v, peak_w = peaks[i]
            gain = peak_w - ctx.feature_priority(i, cur[i])
            if gain <= 1e-6:
                continue
            c_i = float(coeffs_all[i])
            dh = c_i * (peak_v - float(cur[i]))

            compensators = []
            for j in actionable:
                if j == i:
                    continue
                c_j = float(coeffs_all[j])
                v_j = ctx.clip(j, float(cur[j]) - dh / c_j)
                cost = ctx.feature_priority(j, cur[j]) - ctx.feature_priority(j, v_j)
                if cost < gain:
                    compensators.append((cost, j, v_j))
            compensators.sort(key=lambda t: t[0])

            accepted = None
            for _cost, j, v_j in compensators[:max_compensators]:
                cand = cur.copy()
                cand[i] = peak_v
                cand[j] = v_j
                pred = _prediction(ctx.model, cand)
                info["model_calls"] += 1
                residual = pred - float(ctx.target)
                if abs(residual) > float(ctx.epsilon):
                    # One restoration step on the compensating feature, using
                    # the residual measured against the real model.
                    cand[j] = ctx.clip(j, float(cand[j]) - residual / float(coeffs_all[j]))
                    pred = _prediction(ctx.model, cand)
                    info["model_calls"] += 1
                    residual = pred - float(ctx.target)
                    if abs(residual) > float(ctx.epsilon):
                        logger.debug(
                            "[Stage 7] %s -> peak, compensated by %s: "
                            "rejected, |pred - target|=%.4f > epsilon.",
                            ctx.feature_label(i),
                            ctx.feature_label(j),
                            abs(residual),
                        )
                        continue
                cand_priority = ctx.priority_benefit(list(cand))
                if cand_priority is None or cand_priority <= cur_priority + 1e-9:
                    continue
                accepted = (cand, cand_priority, pred, j)
                break

            if accepted is None:
                continue
            cand, cand_priority, pred, j = accepted
            logger.info(
                "[Stage 7] Accepted: %s -> peak %.4f, compensated by %s "
                "-> %.4f | priority %.4f -> %.4f | pred=%.4f.",
                ctx.feature_label(i),
                peak_v,
                ctx.feature_label(j),
                float(cand[j]),
                cur_priority,
                cand_priority,
                pred,
            )
            cur, cur_priority = cand, cand_priority
            accepted_in_sweep += 1
            info["accepted_moves"] += 1
            info["final_prediction"] = pred

        if accepted_in_sweep == 0:
            break

    info["priority_after"] = cur_priority
    info["priority_gain"] = float(cur_priority - info["priority_before"])
    logger.info(
        "[Stage 7] Done: %d move(s) accepted over %d sweep(s) | "
        "priority %.4f -> %.4f (+%.4f) | model calls=%d.",
        info["accepted_moves"],
        info["sweeps_run"],
        info["priority_before"],
        cur_priority,
        info["priority_gain"],
        info["model_calls"],
    )
    return list(cur), info
