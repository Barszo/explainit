# Investigation: MINLP finds no counterfactuals while random search finds them for every sample

## 1. What the results actually show

From `priority_methods/results/diabetes/set1/`:

- `metrics_summary.csv`
  - `minlp`: `n_cfs_total=0`, `n_cfs_valid=0`, `sample_validity_rate=0.0`, `avg_time_seconds≈0.17s`.
  - `random_search`: `n_cfs_valid=40/40`, `sample_validity_rate=0.8` (8/10 samples), `avg_time_seconds≈9s`.
- `counterfactuals.csv`: **every** `minlp` row has
  - `cf_prediction` empty, `validity=False`,
  - `error = failure_reason = "There are no elements fulfilling the requirements"`,
  - `time_seconds ≈ 0.17` (i.e. it fails almost immediately, long before any optimization loop).

So MINLP is **not** "failing to reach the target after searching". It is raising an exception in a **pre-search filtering step**, in ~0.17s, identically for all 10 samples. The 2 samples where random search also failed (`27`, `50`) are a separate, milder issue (random sampling could not hit the target within `max_iterations`), and are not the focus here.

## 2. How the MINLP explainer works (relevant pipeline)

Entry point: `MINLSearchExplainer.find_counterfactuals()` (`explainit/explainers/minlp_search.py`). It runs staged setup **before** the iterative refinement loop:

1. **Stage 0 - derive bounds from priority functions** (`_derive_bounds_and_intervals_from_priorities`, lines ~228-289).
   For each actionable numerical feature it samples the priority function `f(x)` on a grid over the dataset range `[dmin, dmax]`, and computes `allowed_intervals` = the sub-ranges where `f(x) > 0`. It then **narrows** the feature's search bounds:
   - `cfg["min"] = max(dmin, first_positive_x)`
   - `cfg["max"] = min(dmax, last_positive_x)`
   - `cfg["allowed_intervals"] = intervals`
   - If a feature has **no** positive-priority region it raises `ValueError(... Cannot derive feasible bounds)`.

2. **Stage 1 - find a "target exemplar"** (`_stage1_find_exemplar` -> `find_closest_elem` -> `get_rows_in_priorities`, lines ~1641, ~419, ~300-413).
   This is the critical step. `get_rows_in_priorities()` filters the **training dataset rows** down to those that satisfy *all* priority constraints simultaneously:
   - **Categorical**: drop rows whose one-hot combo maps to `None` (e.g. `sex` is `NON_ACTIONABLE`, so all rows with the *other* sex are dropped).
   - **Numerical, applied cumulatively feature-by-feature**: keep only rows with
     `min_val <= row[idx] <= max_val` **and** `row[idx]` inside that feature's `allowed_intervals`.
   - After *each* feature's filter, if `data_np.size == 0` it raises:
     ```python
     raise Exception("There are no elements fulfilling the requirements")
     ```
     (lines 380, 399, 409). **This is the exact error we see.**

   If rows do survive, `find_closest_elem` then additionally requires the closest surviving row's *prediction* to be within `target_exemplar_epsilon` (0.10) of the target, else it raises a different `ValueError`.

3. **Stages 2+ - Shapley/LP/SLSQP refinement**: only reached if a target exemplar was found. It linearises around the exemplar every iteration. We never get here.

## 3. Root cause

**MINLP is dataset-anchored: it requires at least one real training row that lies inside the *joint* (intersection of) all per-feature priority-support windows, plus matches the sample's fixed categorical values.** With the diabetes priority sets this joint region is essentially empty, so Stage 1 raises "There are no elements fulfilling the requirements" before any search happens.

Why the joint region is empty for these priority sets (`priority_sets.py`, diabetes `set1`/`set2`), after Stage 0 turns each priority into an "allowed region" over the dataset range:

- `age`: `peak_at=at_sample(offset=0.5)`, `left=None` -> priority is 0 below the peak. Allowed region is roughly `[sample+0.5, dmax]` (a one-sided upper window; for many samples `sample+0.5` is already near/above `dmax`, making it tiny).
- `bmi`: `peak_at=at_sample()`, `left=at_sample(pct=-0.20)`, `right=None` -> allowed region ≈ `[sample-0.20*range, sample]` (a narrow band just below the sample; 0 above the sample).
- `s1..s5` (serum): `peak_at=at_sample(pct=-0.20 or -0.40)`, `left=at_min`, `right=None` -> allowed region ≈ `[dmin, sample-0.20*range]` (a one-sided lower window; 0 above the peak).
- `bp`: peak with `left=at_min`, `right=at_max` -> allowed region ≈ full range (permissive).
- `s6`: `constant_priority(0.1)` -> allowed region = full range (permissive).
- `sex`: `NON_ACTIONABLE` -> only rows with the sample's own sex are kept.

Each window individually may keep a decent fraction of rows, but MINLP needs **one row that satisfies all of them at once**: `age` high, `bmi` just-below-sample, and `s1..s5` all below their per-feature thresholds, and the right `sex`. Across ~400 diabetes training rows the probability of a single real patient sitting in that specific multi-dimensional corner is ~0, so the cumulative filter collapses to an empty set. Because these windows are *sample-relative*, this happens for every one of the 10 samples, which is exactly what we observe.

## 4. Why random search succeeds where MINLP fails

`RandomSearchExplainer.generate_random_samples` (`explainit/explainers/random_search.py`) does **not** require any real dataset row to be jointly feasible. For each candidate it **samples each feature independently** from that feature's own priority distribution (`sample_numeric_value` / uniform within bounds), assembles a brand-new synthetic point, and keeps it if the model prediction is within `epsilon`.

So random search *constructs* points in the joint priority region even though no such point exists in the dataset. That is precisely the region MINLP cannot enter, because it insists on starting from a real "exemplar" row. This structural difference - synthesise vs. anchor-to-dataset - is the whole story behind "random search finds CFs for all, MINLP finds none".

(Note the CFs random search returns have `n_changed=9/11` and low priority scores for some samples, i.e. they are valid-by-prediction but change almost every feature - a separate quality question, not part of this failure.)

## 5. Secondary gate (would bite even if Stage 1 filtering passed)

Even if a jointly-feasible row existed, `find_closest_elem` requires its prediction to be within `target_exemplar_epsilon = 0.10` of the target. The targets here are `prediction - 0.3`, a large shift; a priority-feasible row that also happens to predict within 0.10 of that shifted target is a second, independent constraint. Keep this in mind: relaxing only the filtering may just move the failure to this check.

## 6. Evidence I could not extract from the current logs/results (please re-run after adding logging)

The results only store the final exception string. They do **not** tell us *which* filter emptied the set or *how restrictive each feature is*. To confirm the diagnosis quantitatively and decide how to relax priorities, I recommend adding the following instrumentation (I have **not** made these changes):

1. **Per-feature and joint coverage, per sample.** There is already a ready-made helper: `explainit/utils/priorities_analyser.py :: compute_priority_coverage(priorities, dataset, feature_names)`. It returns, per feature, `allowed_points_pct` (share of dataset rows inside that feature's allowed region) and a **`global_allowed_pct`** (share inside the joint intersection). My hypothesis predicts `global_allowed_pct ≈ 0` for every sample, while most single features are well above 0.
   - Suggested: in `runner.py`, right after `priorities = build_priorities(...)`, call `compute_priority_coverage` and write a new `minlp_feasibility.csv` with columns: `sample_id`, per-feature `allowed_points_pct`, `global_allowed_pct`, `n_rows`. This alone should confirm the root cause.

2. **Which filter stage empties the set.** In `get_rows_in_priorities` the code already logs `[Stage 1.1a]`/`[Stage 1.1b]` row counts via the `explainit` logger, but nothing records the count *after each numerical feature* nor which feature dropped it to 0. Suggested: capture the surviving row count after every feature filter (categorical, then each numerical idx) and surface it (either persist to the feasibility CSV, or have `MINLPMethod.generate_many` catch the exception and attach a structured `extra={"empty_after_feature": <name>, "rows_after": {...}}` that the runner writes into `counterfactuals.csv`).

3. **Distinguish the two MINLP failure modes.** Currently both "empty filter set" and "no exemplar within `target_exemplar_epsilon`" surface only as raw exception text. Suggested: have `MINLPMethod.generate_many` categorise the failure (e.g. `stage1_empty_filter`, `stage1_exemplar_too_far`, `search_failed`, `not_reached`) into a `failure_stage` column so the progress log makes the pattern obvious at a glance.

If you add (1) and (3) and re-run, the report can be upgraded from "mechanism explained" to "quantified per sample".

## 7. Possible fixes (for your decision - not implemented)

Ordered roughly from "least invasive / changes experiment semantics least" to "structural changes to MINLP".

**A. Relax the priority sets so a jointly-feasible dataset row exists.**
   Widen the restrictive one-sided windows (`age`, `bmi`, `s1..s5`) using the existing relaxation knobs documented in `priority_sets.py`: make `peak_pct`/`pct` less negative, raise `peak_value`, lower exponential `a`, and/or add a `right`/`left` decay instead of hard `None` cutoffs so the support is wider. This keeps MINLP's current design but weakens the strictness of the preferences. Fastest to try; use the coverage helper (Section 6.1) to tune until `global_allowed_pct > 0`.

**B. Stop requiring the exemplar to satisfy the full joint priority region.**
   Decouple exemplar selection from priority-support filtering: in Stage 1, filter only by hard constraints (categorical actionability + explicit user min/max), or don't filter at all, and pick the row closest in prediction to the target. Let Stage 0's `allowed_intervals` + the SLSQP `_project_to_allowed_intervals`/interval constraints (already implemented, lines ~1756-1765, 2190-2199) enforce the priorities *during optimisation* instead of at the exemplar stage. This preserves strict priorities on the final CF while removing the impossible "real row in the joint region" precondition.

**C. Synthesise the exemplar instead of requiring a dataset row.**
   Build the starting anchor by projecting the sample itself into each feature's `allowed_intervals` (the helper `_project_to_allowed_intervals` already exists), rather than searching `get_rows_in_priorities`. This guarantees a feasible start point without depending on the dataset's density.

**D. Soft filtering / nearest-feasible fallback.**
   If `get_rows_in_priorities` empties, fall back to the row that satisfies the *most* feature windows (or is closest to the joint region), instead of raising. Combined with the optimiser's interval constraints, the search can still pull it into the feasible region.

**E. Loosen the second gate.**
   Increase `target_exemplar_epsilon` (config `params.target_exemplar_epsilon`, currently 0.10). This only matters after Stage 1 filtering is fixed; on its own it will not help because we fail earlier.

My recommendation: try **A** first (cheapest, and it directly tests the diagnosis via the coverage helper), and treat **B/C** as the principled longer-term fix if you want MINLP to keep working under strict, sample-relative priorities where no real patient sits in the joint region.

## 8. Suggested next step

Add the logging/saving from Section 6 (items 1 and 3 are enough), **re-run the experiment yourself**, and share `minlp_feasibility.csv`. That will confirm `global_allowed_pct ≈ 0` per sample and show exactly which features are the binding constraints, so you can decide between relaxing priorities (A) or changing MINLP's exemplar logic (B/C).
