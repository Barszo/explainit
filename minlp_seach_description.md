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
- `MINLP COUNTERFACTUAL SEARCH DONE | model_pred(cf)=...`

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

`minlp_search.py` finds a nearby exemplar, explains the gap with Shapley values, turns that into a simpler linear search problem, optimises inside that surrogate, and only afterwards checks what the real model thinks about the returned counterfactual.
