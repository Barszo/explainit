# `minlp_search.py` — simple step-by-step description

## Big picture

`MINLSearchExplainer` tries to answer:

> "How can I change this sample so the model prediction gets close to a target value, while staying inside allowed bounds and following my feature preferences?"

It does this in **two layers**:

1. it uses the **real model** to pick a good reference point (`target_exemplar`),
2. then it builds a **simpler linear approximation** of the model with Shapley values and searches inside that approximation.

That is why logs often show **two different prediction-like numbers**:

- `model_pred(cf)` = the real model output
- `h(x)` / `constraint` = the Shapley-based approximation used during optimisation

If they differ, the approximation was not perfect.

When the surrogate misfires, the algorithm does **not** give up after one shot. Stages 3-6 are wrapped in a **refinement loop**: the candidate from one pass becomes the new "sample" for the next pass, so the surrogate is rebuilt around a point that is already closer to the answer. See *Iterative refinement loop* below for the stopping rules.

---

## Very small example

Imagine a sample has:

- current prediction: `0.36`
- target: `0.75`

The algorithm roughly says:

1. find a real training row whose prediction is already near `0.75`,
2. compare that row to the current sample,
3. estimate which features explain the difference,
4. turn that estimate into a linear equation,
5. solve for feature values that should hit the target,
6. If there are any categorical features - it will calculate this for all combinations present in exemplar and initial sample
7. score the candidates with the priority functions and keep the best one.

---

## Step 1 — Find a target exemplar

### What happens

The code scans the dataset and picks a real row whose prediction is closest to the requested target.

### Why

The method needs a concrete "reference destination" before it can estimate feature contributions.

### Related logs

- `Filtering dataset ... using priority constraints`  
  First it removes rows that violate hard bounds or forbidden categorical choices.
- `Closest filtered-row prediction is ... away from target`  
  Tells you how close the best real row is.
- `Target exemplar locked in ...`  
  Confirms the chosen reference row.

### Simple example

If target is `0.75` and the filtered dataset has predictions:

- `0.40, 0.61, 0.747, 0.89`

then `0.747` becomes the exemplar.

---

## Step 2 — Read the numerical bounds

### What happens

For each numerical feature, the explainer loads `(min, max)` from the priorities.

### Why

These are the hard limits for search. The solver is not supposed to move outside them.

### Related logs

- `Bounds for numerical features: {...}`
- `Non-actionable feature indices ...`

If a feature is non-actionable, it is effectively frozen.

### Simple example

If feature `age` has bounds `[20, 60]`, the search may move it inside that interval only.

---

## Step 3 — Compute Shapley values

### What happens

The code compares:

- the original sample
- the chosen target exemplar

and computes Shapley values for each feature.

### Why

This gives a simple answer to:

> "Which features seem to explain the prediction gap between the sample and the exemplar?"

### How to read them

- positive Shapley value: that feature tends to push prediction upward toward the exemplar side
- negative Shapley value: that feature tends to push downward
- near zero: little effect in this comparison

### Related logs

- `Consolidating features for Shapley ...`
- `Approximate Shapley for unit ...`
- `Shapley values (numerical): {...}`
- `Shapley values (categorical groups): {...}`

If there are no categorical groups, the categorical part is empty.

### Simple example

If the sample predicts `0.36` and the exemplar predicts `0.75`, and Shapley says:

- `income: +0.20`
- `debt: -0.05`
- `savings: +0.10`

then the algorithm reads this as: income and savings helped raise the prediction, debt worked against it.

---

## Step 4 — Build initial feasible solutions

This stage is a bridge between the real model and the optimisation.

### Step 4.1 — Reduce categorical combinations

If a categorical group has zero Shapley effect, the code keeps only the sample's current category.

### Why

This shrinks the search space.

### Log

- `Pruning categorical combinations using Shapley values`
- `X categorical combination(s) survive`

If you have no categorical features, this usually becomes just one empty combination: `{}`.

### Step 4.2 — Cache baseline prediction

The explainer stores the real model prediction for the original sample.

### Why

That prediction is the starting point for the linear approximation.

### Log

- `basic_prediction=...`

### Step 4.3 — Convert Shapley values into a linear target problem

The code builds a linear surrogate:

`h(x) = basic_prediction + numerical_changes + categorical_changes`

For numerical features it converts Shapley values into per-unit coefficients.

### Why

A linear problem is much easier to solve than the original model directly.

### Log

- `LP targets per categorical combo: {...}`

This means: "for this categorical setup, what numerical target must the linear equation satisfy?"

### Step 4.4 — Keep only actionable numerical features

Frozen features are removed from the LP variables.

### Log

- `X actionable numerical features: [...]`

### Step 4.5 — Solve LP for each categorical combination

For each surviving categorical combination, the code solves a bounded linear program to get an initial numerical vector.

### Why

This is a warm start for the next optimisation stage.

### Related logs

- `Combo ... -> LP target=...`
- `LP feasible/infeasible`
- `sanity check: linearised h(x)=...`

Important: this sanity check is about the **linear surrogate**, not the real model.

### Why solutions often sit on bounds

LPs commonly end up on edges or corners of the allowed box, especially when many solutions are possible.

That is why you may see values equal to min or max bounds.

### Simple example

Suppose the surrogate becomes:

`0.36 + 0.10*x1 + 0.05*x2 = 0.75`

The LP may choose one extreme valid solution such as:

- `x1 = max bound`
- `x2 = min bound`

if that still satisfies the equation.

### Step 4.6 — Count how many combinations produced a warm start

### Log

- `1/1 categorical combination(s) yielded an initial feasible solution`

This means stage 5 has something to optimise.

---

## Step 5 — Run SLSQP optimisation

### What happens

Starting from the LP solution, the code runs `scipy.optimize.minimize(..., method="SLSQP")`.

It keeps the bounded linear constraint:

- `target - epsilon <= h(x) <= target + epsilon`

and optimises the priority score.

### Why

The LP only finds "some feasible point".  
SLSQP tries to find a point that is better according to the priority functions.

### Related logs

- `Initial numerical x0=...`
- `Running SLSQP ...`
- `SLSQP done: success=... | model_pred(cf)=... | constraint h(x)=... | objective(weight)=...`

### Important interpretation

At this point:

- `constraint h(x)` tells you whether the **surrogate** thinks the candidate is on target
- `model_pred(cf)` tells you what the **real model** says

If `h(x)` is good but `model_pred(cf)` is not, the linear approximation was too optimistic.

That is exactly the main reason you can later see:

- stage 5 looks successful,
- but the final evaluation says `within ±epsilon: False`.

---

## Step 6 — Pick the best counterfactual

### What happens

If there are multiple categorical combinations, the code compares the final candidates and keeps the best one according to `calculate_total_weight`.

If there are no categorical features, there is only one candidate.

### Related logs

- `Among X candidate(s), pick the one ...`

---

## Iterative refinement loop

### What happens

After Step 6 picks a candidate counterfactual `cf`, the algorithm asks the **real model** what it thinks of `cf`. If the prediction is already within `epsilon` of `target`, the search is finished. If not, the loop:

1. uses `cf` as the new "sample",
2. re-runs steps 3 to 6 against the **same** target exemplar from Step 1,
3. evaluates the new `cf` again,
4. compares the new distance to the best distance seen so far,
5. decides whether to continue, stop, or accept what we have.

Steps 1 and 2 (find exemplar, gather bounds) run **only once**. They depend on the original sample and priorities, which never change.

### Why

A single linearisation around the original sample is often optimistic, especially when the model is non-linear. Re-linearising around a point that is already close to the target gives Shapley values that better describe the local behaviour of the model. Each loop is essentially a small Newton-style correction.

### Parameters

These three parameters live on `find_counterfactuals(...)`:

| Parameter | Default | Meaning |
|---|---|---|
| `max_iterations` | `10` | Hard cap on refinement passes. |
| `patience` | `5` | Stop after this many consecutive iterations that do not improve the best distance to target. |
| `return_when_fails` | `True` | If the loop ends without ever reaching `target ± epsilon`: when `True`, the function still returns the best candidate found and logs a warning; when `False`, it returns `None`. |

### Stop reasons

The loop logs exactly one of these `stop_reason` values:

- `target_reached` — `|model_pred(cf) - target| <= epsilon`. This is the success path.
- `max_iterations` — completed all `max_iterations` passes without ever reaching the target.
- `patience_exhausted` — `patience` consecutive iterations failed to beat the running best distance.
- `search_failed` — an iteration raised an internal error (for example, the LP became infeasible). The function returns the best CF seen before the error, if any.

### How "improvement" is measured

For each iteration the algorithm computes:

`distance = |model_pred(cf) - target|`

It tracks the **best distance so far**. An iteration "improves" only when its distance is **strictly lower** than the current best. The patience counter resets to `0` on improvement and increments by `1` otherwise.

The loop **always advances** to the latest `cf`, even when the iteration did not improve. So a non-improving iteration still changes the linearisation point for the next pass; it just consumes one unit of patience.

### Status object

After every call, the explainer exposes:

```
self.last_search_result = {
    "reached_target": bool,
    "distance":       float,    # best |model_pred - target| seen
    "iterations_run": int,
    "stop_reason":    str,      # one of the four values above
    "best_cf":        list | None,
    "history": [
        {"iteration": int, "model_pred": float, "h_x": float,
         "distance": float, "improved_vs_best": bool},
        ...
    ],
}
```

This is the structured equivalent of "with status that the target was/was not reached".

### Related logs

- `MINLP COUNTERFACTUAL SEARCH | target=... | epsilon=... | max_iterations=... | patience=... | shap_approx=... | num_samples=... | return_when_fails=...`
- `############ REFINEMENT ITERATION i/N ############`
- `[Iter i] model_pred(cf)=... | h(x)=... | distance=... | best_so_far=... | improved_vs_best=...`
- `[Iter i] New best CF (distance=...).`
- `[Iter i] No improvement vs best (k/patience).`
- `[Iter i] Target reached within epsilon=...`
- `Stopping: K consecutive iterations without improvement.`
- `[Iter i] Advancing: next iteration's sample = current CF.`
- `MINLP SEARCH DONE | reached_target=... | stop_reason=... | iterations=... | best_distance=...`

### Small worked example

Target = `0.75`, `epsilon = 0.05`, `patience = 5`, `max_iterations = 10`.

| Iter | model_pred(cf) | distance | best so far | action |
|---|---|---|---|---|
| 1 | 0.83 | 0.08 | 0.08 (new best) | advance, continue |
| 2 | 0.79 | 0.04 | 0.04 (new best) | within epsilon, stop |

`stop_reason = "target_reached"`, function returns the iteration 2 CF.

A failure example:

| Iter | model_pred(cf) | distance | best so far | action |
|---|---|---|---|---|
| 1 | 0.83 | 0.08 | 0.08 (new best) | advance, continue |
| 2 | 0.86 | 0.11 | 0.08 | no progress (1/5), advance |
| 3 | 0.85 | 0.10 | 0.08 | no progress (2/5), advance |
| 4 | 0.84 | 0.09 | 0.08 | no progress (3/5), advance |
| 5 | 0.84 | 0.09 | 0.08 | no progress (4/5), advance |
| 6 | 0.84 | 0.09 | 0.08 | no progress (5/5), STOP |

`stop_reason = "patience_exhausted"`. With `return_when_fails=True` (default) the function returns the iteration 1 CF (best distance), logs a warning, and `self.last_search_result` records all six rows in `history`.

---

## What `interactive_minlp_cont.py` adds on top

`interactive_minlp_cont.py` is the workbench around `minlp_search.py`.

After `find_counterfactuals()` returns, it:

1. evaluates the returned counterfactual with the **real model**,
2. measures distance from the target,
3. prints feature changes.

### Related logs

- `COUNTERFACTUAL | y(scaled)=... | target(scaled)=... | gap=...`
- `within ±...: True/False`
- `Validity FAIL ...`
- `Top changed features:`

So these logs are the final reality check.

---

## How to read the logs in general

## If you see this...

### `Target exemplar selected`

Read it as:

> "We found a real reference point near the target."

### `Shapley values ...`

Read it as:

> "These features seem to explain the difference between sample and exemplar."

### `LP feasible`

Read it as:

> "The simplified linear version of the problem has a valid solution."

### `constraint h(x)=target`

Read it as:

> "The surrogate model is satisfied."

### `model_pred(cf)` far from target

Read it as:

> "The real model disagrees with the surrogate."

### `Validity FAIL`

Read it as:

> "The final returned point does not actually hit the requested target closely enough."

### `REFINEMENT ITERATION i/N`

Read it as:

> "Starting refinement pass `i` of `N`. Stages 3-6 will run again with the latest candidate as the new sample."

### `MINLP SEARCH DONE | reached_target=... | stop_reason=...`

Read it as:

> "The whole loop has finished. `reached_target` says whether the real model output ended up inside `epsilon`; `stop_reason` says why the loop ended."

---

## Why the final result can fail even when stage 5 says success

This is the most important thing to understand.

`minlp_search.py` does **not** constrain the real model directly in the continuous flow.  
It constrains the **Shapley-based linear approximation** `h(x)`.

So the pipeline can do this:

1. satisfy `h(x)` perfectly,
2. satisfy SLSQP mathematically,
3. still miss the real target when passed through the neural network.

In short:

> stage 4 and 5 solve the surrogate problem, not the full real-model problem.

---

## One-sentence summary

`minlp_search.py` finds a nearby exemplar, explains the gap with Shapley values, turns that into a simpler linear search problem, optimises inside that surrogate, checks what the real model thinks about the returned counterfactual, and **iteratively re-linearises around the latest candidate** until it lands within `epsilon` of the target, hits the patience or iteration cap, or runs out of feasibility.
