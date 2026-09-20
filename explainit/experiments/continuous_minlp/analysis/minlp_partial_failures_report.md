# Investigation #2: MINLP still misses counterfactuals that random search finds

This follows up on the fallback work. The exemplar fallback fixed the "instant
`There are no elements fulfilling the requirements`" crash, but MINLP still fails
to produce a valid counterfactual (CF) for many samples where random search
succeeds, and it is noticeably worse on `set2` than on `set1`. This report
explains why, using the current results in
`priority_methods/results/diabetes/{set1,set2}/`.

## 1. What the new results show

Aggregate (`metrics_summary.csv`):

| set | method | samples with valid CF | cf validity rate |
|-----|--------|-----------------------|------------------|
| set1 | minlp | 5/10 | 5/9 |
| set1 | random_search | 8/10 | 40/40 |
| set2 | minlp | 2/10 | 2/8 |
| set2 | random_search | 5/10 | 25/25 |

Two facts jump out of `counterfactuals.csv`:

1. **Every MINLP row has `exemplar_source = dataset_actionable`.** The primary
   strategy (`dataset_priority_filtered`) still fails for all samples (the joint
   priority region is still empty, exactly as in the first report), and the
   `random_search` fallback is **never** used. So the anchor MINLP now optimises
   against is always a real dataset row whose non-actionable features are frozen
   at the sample - a row that does **not** lie inside the actionable features'
   priority windows.

2. **MINLP no longer crashes; instead its refinement loop fails to converge.**
   The `failure_reason` values are now `MINLP did not reach target:
   {search_failed | max_iterations | patience_exhausted}` or `no counterfactual
   produced` - i.e. the Shapley/LP/SLSQP optimisation ran but did not get within
   `epsilon` of the target.

## 2. Per-sample breakdown (random search as a feasibility oracle)

Random search samples points directly from the *same* allowed region and keeps
any point within `epsilon`. So "random search found a CF" is strong evidence
that **a feasible point exists in the priority region**; if random search fails
after its 10k iterations, the region almost certainly cannot reach the target.
Using that to classify each MINLP failure:

| sample | set1 random | set1 minlp | set1 verdict | set2 random | set2 minlp | set2 verdict |
|--------|-------------|------------|--------------|-------------|------------|--------------|
| 60 | valid | valid | ok | **fail** | patience_exhausted | **feasibility** |
| 2  | valid | valid | ok | valid | max_iterations | **convergence** |
| 52 | valid | search_failed | **convergence** | **fail** | no cf | **feasibility** |
| 18 | valid | valid | ok | valid | patience_exhausted | **convergence** |
| 27 | **fail** | search_failed | **feasibility** | **fail** | search_failed | **feasibility** |
| 26 | valid | valid | ok | valid | patience_exhausted | **convergence** |
| 38 | valid | valid | ok | valid | valid | ok |
| 50 | **fail** | no cf | **feasibility** | **fail** | no cf | **feasibility** |
| 24 | valid | max_iterations | **convergence** | **fail** | search_failed | **feasibility** |
| 33 | valid | patience_exhausted | **convergence** | valid | valid | ok |

This cleanly separates the user's three questions:

- **Sample 27 (both sets, neither method):** genuine **feasibility** limit. No
  point in the priority-allowed region predicts within `epsilon=0.05` of the
  target, so nothing can succeed. This is the "expected" case.
- **Sample 26 (minlp set1 yes, set2 no; random both yes):** a feasible point
  exists in both sets (random finds it), so set2 is a **convergence** failure -
  MINLP's local optimiser got stuck (`patience_exhausted`) on set2's tighter
  landscape even though a valid CF exists.
- **Sample 24 (minlp neither; random set1 yes, set2 no):** set1 is a
  **convergence** failure (feasible, MINLP hit `max_iterations`); set2 is a
  **feasibility** failure (random search also fails, so no reachable point).

## 3. Root causes

### 3a. The `dataset_actionable` fallback anchors the search *outside* the feasible region
In the original design the exemplar was a **priority-filtered** row, i.e. a point
already inside every feature's allowed interval. MINLP then computes Shapley
values between the sample and that exemplar and linearises around it. The
`dataset_actionable` fallback deliberately drops the priority-window filtering,
so the anchor now sits **outside** the allowed region for the actionable
features. Linearising a highly non-linear neural network around a point that the
constrained optimiser can never reach yields poor search directions, which is a
direct cause of the `patience_exhausted` / `max_iterations` stalls. This is a
quality regression that only shows up now that the fallback is always active.

### 3b. The strong `random_search` fallback is effectively dead code
The fallback chain is: primary -> `dataset_actionable` -> `random_search`. But
`_find_exemplar_actionable_fallback` has **no acceptance test** - it always
returns the closest available row (`argmin` over distances), so it never raises,
so the `random_search` fallback is never reached (confirmed: `exemplar_source`
is `dataset_actionable` for all 20 MINLP runs). The random-search fallback,
which samples the allowed region and only accepts a point **within `epsilon`**,
is exactly what would help the convergence cases - yet it never runs.

Even if it did run, note a second gap: the random-search fallback only sets the
*anchor* (`target_exemplar`); the value it finds is already a valid CF, but MINLP
does not return it directly - it still runs the refinement loop and returns
`best_cf` from that loop, which can be worse. So a within-`epsilon` point found
by the fallback can still be discarded.

### 3c. MINLP is a local optimiser on a non-convex model with narrow interval constraints
`find_counterfactuals` runs Shapley -> LP warm-starts -> SLSQP under
`allowed_intervals` constraints (see `_run_one_pass`; the SLSQP interval
constraints are added around lines ~1756-1765). With the diabetes priorities the
allowed regions are narrow, one-sided, and can be split into disjoint intervals.
On such a landscape:
- `search_failed` = an LP/SLSQP pass raised (e.g. an infeasible warm-start or an
  SLSQP failure) and the loop aborted;
- `patience_exhausted` / `max_iterations` = the local search made no useful
  progress within the tight constraints.

Random search does not care about gradients or convexity: it does global
rejection sampling, so it routinely lands feasible points that MINLP's local
search cannot navigate to. That is the fundamental reason random search wins
wherever the region is feasible.

### 3d. Why set2 is worse than set1
`set2` uses strictly tighter serum priorities than `set1`:
`_diabetes_serum_priorities(peak_pct=-0.40, peak_value=0.5)` vs
`(peak_pct=-0.20, peak_value=0.7)` (see `priority_sets.py`). Peaks sit further
below the sample with a lower height, so the allowed serum region is smaller and
shifted further from the sample. That has two compounding effects:
- **more feasibility-empty samples** (60, 52, 24 flip from feasible in set1 to
  infeasible in set2), and
- **a harder optimisation landscape** for the samples that are still feasible
  (2, 18, 26 are feasible but MINLP stalls). Hence set2's MINLP validity drops to
  2/10.

## 4. What the current logs still cannot tell us (please add, then re-run)

The results now record `exemplar_source`, which was enough to reach the diagnosis
above, but two blind spots remain. I have **not** made these changes.

1. **Feasibility vs convergence should be recorded directly, not inferred.**
   Right now I infer it from whether the separate `random_search` *method* found
   a CF. Add an explicit per-(sample, set) feasibility probe: sample the allowed
   region (the existing `_find_exemplar_random_fallback` / random logic) and
   record `feasible_within_epsilon` (bool) and the `min_pred_distance` achieved.
   Persist it next to the results (e.g. `minlp_feasibility.csv`). This turns the
   inferred table in Section 2 into measured columns.

2. **Preserve the granular MINLP stop reason and anchor quality even on total
   failure.** When `best_cf` is `None` (samples 50, and 52/24 on set2) the row
   only says `no counterfactual produced`; the underlying `stop_reason`
   (`search_failed` vs `max_iterations`) and `iterations_run` are dropped by the
   runner's `cf is None` branch. Also log the fallback exemplar's own
   `|pred - target|` (returned by `_find_exemplar_actionable_fallback`) so we can
   see how bad the anchor was. Optionally capture the SLSQP/LP exception text for
   `search_failed`. These make the convergence failures debuggable without
   re-instrumenting each run.

## 5. Possible fixes (for your decision - not implemented)

Grouped by the failure class they address.

**For convergence failures (feasible, but MINLP stalls): 52/24/33 set1, 2/18/26 set2**

- **F1. Gate the `dataset_actionable` fallback so `random_search` can trigger.**
  Accept the actionable exemplar only if its `|pred - target|` is within
  `target_exemplar_epsilon`; otherwise cascade to the random-search fallback.
  This makes the strong fallback reachable exactly when the dataset anchor is
  poor.
- **F2. Let the random-search fallback short-circuit to a returned CF.** When it
  finds a within-`epsilon` point, seed `best_cf` with it (or return it directly
  if the refinement fails to beat it). This guarantees MINLP is at least as good
  as random search wherever the region is feasible.
- **F3. Anchor inside the feasible region.** Project the exemplar into each
  feature's `allowed_intervals` (helper `_project_to_allowed_intervals` already
  exists) before computing Shapley, so the linearisation is taken at a reachable
  point.
- **F4. Add restarts / raise budget.** Multi-start SLSQP from several exemplars
  (or several random-feasible seeds), and/or increase `max_iterations` /
  `patience` in `config.yaml`. Cheap to try, helps the local-stall cases.

**For feasibility failures (no reachable point): 27/50 (both sets), 60/52/24 set2**

- **F5. Relax the priorities** (documented knobs in `priority_sets.py`: less
  negative `peak_pct`, higher `peak_value`, lower exponential `a`, or add a
  decaying `left`/`right` instead of hard `None` cutoffs), and/or **raise
  `epsilon`**. This is the only way to make these samples solvable by *any*
  method; set2 needs it more than set1.
- **F6. Detect and report infeasibility as a first-class outcome** (via the
  Section 4.1 probe) so these samples are labelled "infeasible under priorities"
  rather than looking like a method bug.

My recommendation: implement **F1 + F2** first (they directly convert the
convergence losses into wins by actually using the random-search fallback that
already exists), consider **F3** for anchor quality, and treat the feasibility
cases with **F5/F6**. Add the Section 4 logging and re-run to confirm the
feasibility-vs-convergence split as measured columns.

## 6. One-paragraph summary

MINLP no longer crashes because the `dataset_actionable` fallback always finds an
anchor - but that anchor sits outside the priority region, so the local
Shapley/LP/SLSQP optimiser frequently stalls (`patience_exhausted`,
`max_iterations`, `search_failed`), while the stronger `random_search` fallback
never triggers because the actionable fallback never "fails". Where a feasible
point exists (random search proves it), these are **convergence** failures MINLP
should be able to win with F1-F4; where random search also fails (sample 27, 50,
and several set2 samples), the priority region genuinely cannot reach the target
(**feasibility**), solvable only by relaxing priorities/epsilon (F5). set2 is
worse than set1 purely because its tighter serum windows shrink the feasible
region on both axes at once.
