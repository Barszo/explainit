# Counterfactual explainers tutorial: Random Search vs. MINLP

This document explains the two counterfactual generation methods implemented in this repo:

- `RandomSearchExplainer` (`explainit/explainers/random_search.py`)
- `MINLSearchExplainer` (`explainit/explainers/minlp_search.py`)

It focuses on **practical usage** (how to set priorities, how to run the example scripts, how to debug),
while also including a **scientific/mathematical view** for advanced readers.

---

## 1) What is a counterfactual explanation?

Given:

- a model \(f:\mathbb{R}^d \rightarrow \mathbb{R}\) (regression) or \(f:\mathbb{R}^d \rightarrow [0,1]\) (probability),
- an input (the “factual”) \(x \in \mathbb{R}^d\),
- a target behavior (e.g. “flip class”, or “reach prediction \(t\)”),

a **counterfactual** is a new input \(x'\) such that:

1. **Validity**: the model output meets the goal (e.g. \(f(x') > \tau\) for class 1, or \(|f(x') - t| \le \varepsilon\)),
2. **Actionability / feasibility**: changes respect domain constraints (bounds, forbidden categories),
3. **Preference / plausibility**: among valid candidates, pick those that best satisfy user preferences.

In this repo, **preferences are expressed via a `priorities` dictionary** (details below).

---

## 2) The `priorities` dictionary (core concept)

Both explainers take a `priorities` dict with two top-level keys:

```python
priorities = {
  "numerical": { ... },
  "categorical": { ... },
}
```

### 2.1 Numerical (continuous) priorities

Numerical priorities map a **feature index** to either:

- an **actionable constraint**:

```python
priorities["numerical"][idx] = {
  "min": <hard lower bound>,
  "max": <hard upper bound>,
  "function": f,  # preference weight in [0,1]
}
```

where `f(x)` should be a preference weight in \([0,1]\):

- `1.0` = highly preferred,
- `0.0` = not preferred (but still potentially allowed, unless you also restrict bounds).

Or a **non-actionable marker**:

- `RandomSearchExplainer`: `0` or `None` means “fixed to the original sample value”.
- `MINLSearchExplainer` (new format): `{"min": sample[idx], "max": sample[idx], "function": None}` is the explicit “fixed” representation.

#### Preference function helpers used in examples

The examples use helper curves from:

- `explainit/priorities/linear.py` → `basic_linear(x, x0, x1, increasing)`
- `explainit/priorities/nonlinear.py` → `exponential(x, x0, x1, increasing, a)`

Interpretation (for both helpers):

- With `increasing=True`: weight transitions from 0 → 1 as \(x\) goes from \(x_0\) to \(x_1\)
- With `increasing=False`: weight transitions from 1 → 0 as \(x\) goes from \(x_0\) to \(x_1\)

This lets you encode “prefer small values” or “prefer large values” without hard-forbidding them.

### 2.2 Categorical priorities

Categorical priorities map a **group of indices** (tuple) to a table:

```python
priorities["categorical"][(i, j, ...)] = {
  (cat_i, cat_j, ...): weight,
  ...
}
```

Examples:

- a single categorical feature at index 12: key is `(12,)`, value keys look like `(3,)`, `(6,)`, …
- a grouped decision with two indices `(8, 10)`: value keys look like `(0, 2)`, `(1, 3)`, …

**Random search convention**

- weight `0` means **forbidden**; the explainer filters such entries out before sampling.

**MINLP convention**

- numeric weights are used as part of the objective (preference score).
- some parts of the MINLP implementation treat `None` as “forbidden” during dataset filtering (`get_rows_in_priorities` removes rows matching `None` category combos).

Practical rule:

- For random search: use `0` to forbid.
- For MINLP: you can use `None` to forbid (filters the reference dataset), but keep the sample’s current category allowed or search can become infeasible.

---

## 3) RandomSearchExplainer (sampling-based)

### 3.1 Practical intuition

Random search is a baseline that works when you can afford many random trials:

1. Sample candidate \(x'\) values inside your bounds / allowed categories.
2. Evaluate the model \(f(x')\).
3. Keep candidates that satisfy the goal.
4. Rank kept candidates by preference score.

It is simple, robust, and easy to debug (you can inspect candidates and why they were rejected).

### 3.2 How sampling works in this repo

In `explainit/explainers/random_search.py`:

- Numerical sampling uses rejection sampling:
  - propose \(u \sim \text{Uniform}(\text{min}, \text{max})\)
  - accept with probability \(f(u)\)
  - retry up to `max_tries`, then fall back to a discretized “sample proportional to \(f\)” approach.
- Categorical sampling:
  - creates a list of allowed combinations (weight > 0),
  - samples a combination with probability proportional to its weight.

### 3.3 Validity conditions (binary vs. continuous target)

RandomSearchExplainer exposes two main entry points:

- **Binary classification**: `generate_for_binary(...)`
  - valid if `pred > threshold` (target_class=1) or `pred < threshold` (target_class=0)
- **Continuous target**: `generate_random_samples(..., epsilon=...)`
  - valid if \(|pred - target| \le \varepsilon\)

### 3.4 Preference scoring in this repo

The current implementation computes an additive score (higher is “better”):

\[
S(x') = \sum_{i \in \text{numerical actionable}} w_i(x'_i) \;+\; \sum_{g \in \text{categorical groups}} W_g(x'_g)
\]

Where:

- \(w_i(\cdot)\in[0,1]\) is your numerical preference function,
- \(W_g(\cdot)\in[0,1]\) is your categorical table weight for group \(g\).

You can inspect contributions with `explainer.get_preference_breakdown(cf)`.

### 3.5 Important implementation detail (regression use)

The generator constructs each candidate from an all-zeros vector and fills only indices present in priorities.
If you forget to cover an index, it may be forced to `0` (unintended and often invalid).

That’s why the continuous tutorial script sets:

```python
numerical = {i: 0 for i in range(n_features)}  # fix everything by default
# then it replaces a few indices with actionable dicts
```

### 3.6 Example scripts for Random Search

#### Binary tutorial

File: `explainit/examples/binary_random_search_example.py`

What it does:

- downloads Heart Disease dataset
- trains `LogisticRegression` (scikit-learn)
- auto-selects a sample and chooses `target_class`
- builds numerical + categorical priorities
- runs `generate_for_binary(...)`
- prints changes + `get_preference_breakdown(...)`

Run it:

```bash
python3 explainit/examples/binary_random_search_example.py
```

Where to edit:

- `build_priorities(...)`: change bounds and preference directions, forbid/allow categories.
- `max_iterations`, `max_tries`, `n_candidates_per_cf`: trade speed vs. chance of finding CFs.

#### Continuous (regression) tutorial

File: `explainit/examples/continuous_random_search_example.py`

What it does:

- downloads California Housing
- scales X and y to \([0,1]\) for readability
- trains `GradientBoostingRegressor`
- chooses a sample and a continuous target `target` with tolerance `epsilon`
- demonstrates:
  - continuous priorities via `linear.basic_linear` and `nonlinear.exponential`
  - categorical priorities via a derived categorical feature `MedIncBand`
- runs `generate_random_samples(..., epsilon=...)`

Run it:

```bash
python3 explainit/examples/continuous_random_search_example.py
```

Notes:

- It sets `explainer.n_candidates_per_cf = 3` because `generate_random_samples` reads that attribute internally.
- It will save priority plots under `images/` if matplotlib is available.

---

## 4) MINLSearchExplainer (optimization-based, Shapley-guided)

### 4.1 Practical intuition

MINLP is meant to be **more directed** than random search.
Instead of blind sampling, it tries to:

1. Find a **target exemplar** \(z\) from a reference dataset \(D\) such that \(f(z)\) is close to the desired target.
2. Compute **Shapley values** that estimate which features matter for moving from \(x\) to \(z\).
3. Solve an optimization problem over actionable variables that satisfies the target constraint and maximizes preferences.

It is often more compute-heavy per run than random search, but can explore a smaller, more structured space.

### 4.2 Target exemplar selection

The method `get_rows_in_priorities()` filters the dataset to rows that obey your hard constraints:

- categorical: removes rows matching forbidden (`None`) category combos
- numerical: keeps rows inside `[min, max]`

Then `find_closest_elem()` selects \(z\) such that:

\[
z = \arg\min_{d \in D_{\text{filtered}}} |f(d) - t|
\]

with a requirement that the closest distance is within `target_exemplar_epsilon`.

### 4.3 Shapley values (advanced view)

Conceptually, Shapley values allocate the prediction difference between a “baseline” and a “target” across features:

\[
f(z) - f(x) \approx \sum_{i=1}^d \phi_i
\]

where \(\phi_i\) is the Shapley contribution of feature \(i\).

This repo’s implementation also supports grouping categorical features so they can be treated as a single unit for Shapley computation.

You can use:

- exact (combinatorial) Shapley for small numbers of “units”
- approximate Shapley by sampling subsets (see `shap_approx=True` and `num_samples` arguments)

### 4.4 The constraint approximation used by this implementation (advanced view)

The MINLP code constructs coefficients that look like:

\[
a_i = \frac{\phi_i}{z_i - x_i}
\]

and then uses these to build a constraint function that (roughly) steers \(x'\) so that the model output reaches the target.

**Numerical stability note**: if \(z_i \approx x_i\), the denominator can be near-zero.
The implementation was adjusted to treat such coefficients as 0 instead of producing NaNs.

### 4.5 Objective (preference score)

The objective uses your priorities similarly to random search:

- numerical features contribute `function(value)` (when function is not `None`)
- categorical groups contribute their table weight

Internally, `calculate_total_weight(...)` returns the summed weight.

### 4.6 Example scripts for MINLP

#### Binary tutorial

File: `explainit/examples/binary_minlp_search_example.py`

What it does:

- downloads Heart Disease dataset
- trains `LogisticRegression`
- chooses a sample and `target_class` for a thresholded classifier
- builds priorities (numerical new-format + categorical tables)
- runs:
  - optional `get_rows_in_priorities()` to show filtering effect
  - `find_counterfactuals_for_binary(...)`

Run it:

```bash
python3 explainit/examples/binary_minlp_search_example.py
```

Where to edit:

- `build_priorities(...)`: actionable numeric list + bounds + preference directions
- `max_iterations`: optimization iterations
- `shap_approx`: set `True` to speed up Shapley computation

#### Continuous (regression) tutorial

File: `explainit/examples/continuous_minlp_search_example.py`

What it does:

- downloads Wine Quality (red) from UCI (with SSL fallback mirrors)
- scales X and y to \([0,1]\)
- trains a simple `Ridge` regressor
- chooses a sample and a **continuous target** close to the sample’s prediction
- builds priorities for a small set of actionable variables
- runs `find_counterfactuals(shap_approx=True, ...)`

Run it:

```bash
python3 explainit/examples/continuous_minlp_search_example.py
```

Important practical notes for this tutorial:

- The script **reorders feature columns** so that the actionable numerical features occupy indices `0..k-1`.  
  This matches how the current `MINLSearchExplainer.find_counterfactuals` indexes its optimizer vector.
- It uses `shap_approx=True` to keep runtime reasonable.
- It may relax `epsilon` (tolerance) if the first run fails.

---

## 5) When should you use which explainer?

### Use Random Search when…

- You want the simplest method to get running.
- You have a “good enough” preference structure and can run many iterations.
- You want a baseline for comparison with more advanced methods.

### Use MINLP when…

- You have a representative reference dataset in the same feature space.
- You want a more directed search guided by feature importance.
- You can tolerate heavier per-run compute and more implementation-specific constraints.

---

## 6) Troubleshooting checklist

### RandomSearchExplainer returns “No valid candidates”

Common causes:

1. **Target too strict**:
   - for regression: epsilon too small
   - for classification: threshold too aggressive
2. **Priorities too restrictive**:
   - hard bounds exclude feasible solutions
   - forbidden categories eliminate the necessary combos
3. **Regression detail**: you forgot to cover all indices in `priorities["numerical"]`, so some features became 0.

Fixes:

- increase `max_iterations`
- widen epsilon (regression)
- relax min/max bounds or categorical restrictions
- start from a different sample (e.g. one closer to the decision boundary / target)

### MINLSearchExplainer fails early

Common causes:

1. **No target exemplar found** within `target_exemplar_epsilon`
2. **NaNs** due to zero denominators (feature doesn’t change sample→exemplar)
3. **Too many variables** (Shapley becomes expensive)

Fixes:

- increase `target_exemplar_epsilon`
- reduce number of actionable numerical indices
- use `shap_approx=True`
- pick a closer target (for regression) and/or increase epsilon tolerance

---

## 7) Minimal “template” snippets (copy/paste)

### Random search (continuous target)

```python
from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.nonlinear import exponential

sample = X_test[i]
target = 0.55
epsilon = 0.03

def pref_hi(x):  # prefer higher values
    return float(exponential(x, x0=0.4, x1=0.8, increasing=True, a=6))

priorities = {
  "numerical": {
    j: 0 for j in range(X_train.shape[1])  # fix everything by default
  },
  "categorical": {},
}

priorities["numerical"][0] = {"min": 0.0, "max": 1.0, "function": pref_hi}

explainer = RandomSearchExplainer(model_pred=model.predict, priorities=priorities, sample=sample, target=target)
explainer.n_candidates_per_cf = 3
counterfactuals, preds, scores, iters = explainer.generate_random_samples(
    expected_counterfactuals=3,
    max_iterations=20000,
    epsilon=epsilon,
    use_monte_carlo=True,
)
```

### MINLP (continuous target)

```python
from explainit.explainers.minlp_search import MINLSearchExplainer

priorities = {
  "numerical": {
    0: {"min": 0.0, "max": 1.0, "function": my_pref_fn},
    1: {"min": 0.0, "max": 1.0, "function": my_pref_fn2},
  },
  "categorical": {},
}

explainer = MINLSearchExplainer(
  model_pred=model.predict,
  priorities=priorities,
  sample=sample,
  target=target,
  dataset=X_train,
  target_exemplar_epsilon=0.2,
  epsilon=0.1,
)

cf = explainer.find_counterfactuals(shap_approx=True, num_samples=120)
```

---

## 8) Files you should look at

Explainability algorithms:

- `explainit/explainers/random_search.py`
- `explainit/explainers/minlp_search.py`

Priority helper functions:

- `explainit/priorities/linear.py`
- `explainit/priorities/nonlinear.py`

Tutorial scripts (binary + continuous):

- `explainit/examples/binary_random_search_example.py`
- `explainit/examples/binary_minlp_search_example.py`
- `explainit/examples/continuous_random_search_example.py`
- `explainit/examples/continuous_minlp_search_example.py`
