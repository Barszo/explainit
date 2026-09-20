# What "projection" means in MINLP-Search (with numeric examples)

In this workflow, **projection** means:  
for each numerical feature, if a value is outside its allowed priority intervals, replace it with the **nearest allowed point**.

The code does this with:
- membership check: `_in_allowed_intervals(...)`
- snap-to-nearest boundary: `_project_to_allowed_intervals(...)`

---

## Example 1, single allowed interval

Allowed interval for feature $x_i$: $[1.0, 4.0]$

- value $x_i=2.5$ is already allowed, projection keeps $2.5$
- value $x_i=0.2$ is outside, nearest boundary is $1.0$, projected value is $1.0$
- value $x_i=5.7$ is outside, nearest boundary is $4.0$, projected value is $4.0$

---

## Example 2, multiple allowed intervals

Allowed intervals: $[1,2]\cup[5,6]$

- value $3.7$: distances to boundaries are  
  $|3.7-2|=1.7$, $|5-3.7|=1.3$, so projected value is $5$
- value $4.4$: nearest boundary is also $5$, projected value is $5$
- value $2.2$: nearest boundary is $2$, projected value is $2$

So projection can jump across a forbidden gap to the nearest admissible boundary.

---

## Why this mattered for the regression

Before the fix, fallback exemplar selection could accept an exemplar that was good **before projection**, but bad **after projection**.

Numerical example:

- target: $y^\star=0.02$
- actionable fallback exemplar gives prediction $f(x^E)=0.039$  
  gap before projection: $|0.039-0.02|=0.019$ (looks good)
- after projecting feature values into strict allowed intervals, prediction becomes $f(\tilde{x}^E)=0.389$  
  gap after projection: $|0.389-0.02|=0.369$ (very bad)

If this bad projected exemplar is used for Shapley linearisation, LP/SLSQP is guided in the wrong local direction.

---

## What the fixed logic does now

For fallback `dataset_actionable` exemplar:

1. compute gap before projection
2. project exemplar into allowed intervals
3. compute gap after projection
4. accept only if post-projection gap is still small (now checked against final tolerance)
5. otherwise reject and switch to `random_in_range` anchor generation

Additionally, if the final selected exemplar is already within $\epsilon$, it is kept as a safe fallback best candidate.

---

## Tiny pseudo-flow

1. Select exemplar candidate  
2. Project numerical features into allowed intervals  
3. Re-evaluate $ |f(x^E)-y^\star| $  
4. If acceptable, continue with Shapley + LP + SLSQP  
5. If not acceptable, use another fallback exemplar strategy

---

## Full 6-step flow with concrete numbers

Below is one numeric walkthrough matching the current logic:

- target $y^\star = 0.20$
- final tolerance $\varepsilon = 0.05$

### Step 1, try normal priority-filtered dataset exemplar

Suppose after filtering the dataset by full priority constraints, no row is available close enough to target (or filtering becomes empty).  
Result: **Step 1 fails**.

### Step 2, try `dataset_actionable` exemplar

Actionability-only fallback picks a dataset row:
$$
x^E_{\text{actionable}}=(3.2,\;1.7,\;5.1),\qquad f(x^E_{\text{actionable}})=0.215
$$
So before projection:
$$
|0.215-0.20|=0.015
$$
Looks good.

### Step 3, project this exemplar into allowed intervals

Assume allowed intervals are:

- feature 1: $[1.0,2.0]\cup[4.0,4.5]$  
- feature 2: $[1.5,1.9]$  
- feature 3: $[2.0,3.0]\cup[6.0,6.4]$

Projection:

- $3.2 \to 4.0$ (nearest allowed boundary)  
- $1.7 \to 1.7$ (already allowed)  
- $5.1 \to 6.0$ (nearest allowed boundary)

Projected exemplar:
$$
\tilde{x}^E=(4.0,\;1.7,\;6.0)
$$

### Step 4, accept only if post-projection gap is within epsilon

Suppose:
$$
f(\tilde{x}^E)=0.34
$$
Then:
$$
|0.34-0.20|=0.14 > 0.05
$$
So this projected actionable exemplar is **rejected**.

### Step 5, generate random in-range exemplar directly in allowed region

Now sample directly from allowed region until one satisfies target tolerance.

Example sampled candidate:
$$
x^E_{\text{rand}}=(1.8,\;1.6,\;2.7),\qquad f(x^E_{\text{rand}})=0.23
$$
Gap:
$$
|0.23-0.20|=0.03 \le 0.05
$$
This exemplar is **accepted**.

### Step 6, keep valid selected exemplar as fallback CF

Because accepted exemplar already satisfies:
$$
|f(x^E_{\text{rand}})-y^\star|\le\varepsilon
$$
it is stored as a safe fallback best candidate.

So if later Shapley+LP+SLSQP iterations fail, stall, or drift to worse points, the algorithm can still return at least this valid in-range exemplar instead of returning no CF.
