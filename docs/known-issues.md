# Known issues

This page lists confirmed bugs and algorithm limitations in explainit. They are
**deliberately not fixed yet**: the code base is being refactored first, with
tests that guarantee the algorithms behave exactly as before. Fixing these
issues is the next step once the refactoring is complete.

How issues are tracked:

- Every bug marked *Tested* has a test in
  [`tests/test_known_bugs.py`](../tests/test_known_bugs.py) that describes the
  correct behaviour and is marked as an expected failure (`xfail`). Fixing the
  bug makes that test pass, which turns the suite red until the `xfail` marker
  is removed, so a fix cannot go unnoticed.
- Locations refer to function and method names, which survive the
  refactoring better than line numbers. They are updated as code moves.
- When you fix an issue, delete its entry here, remove the `xfail` marker,
  and mention the fix in [`CHANGELOG.md`](../CHANGELOG.md). Fixes that change
  results must also re-record the snapshots (see
  [CONTRIBUTING.md](../CONTRIBUTING.md#types-of-tests)) and include a benchmark
  comparison.

## Summary

| ID | Component | Problem | Impact | Tested |
| --- | --- | --- | --- | --- |
| [MINLP-1](#minlp-1-allowed-ranges-are-ignored) | MINLP | Allowed `min`/`max` ranges are ignored | **High**: answers outside the allowed range | Yes |
| [MINLP-2](#minlp-2-the-callers-priorities-are-modified) | MINLP | The caller's priorities dictionary is modified | **High**: later runs silently use different ranges | Yes |
| [MINLP-3](#minlp-3-the-anchor-can-carry-wrong-immutable-values) | MINLP | The returned anchor can have wrong immutable-feature values | **High**: "unchangeable" features change | Yes |
| [MINLP-4](#minlp-4-binary-search-overwrites-the-explainers-settings) | MINLP binary | Overwrites the explainer's `target` and `epsilon` | Medium | Yes |
| [MINLP-5](#minlp-5-binary-results-are-sorted-worst-first) | MINLP binary | Results are sorted worst-first | Medium | No (needs two results) |
| [MINLP-6](#minlp-6-binary-search-ignores-expected_counterfactuals) | MINLP binary | `expected_counterfactuals` is ignored | Low | No |
| [MINLP-7](#minlp-7-tied-predictions-crash-the-anchor-search) | MINLP | Tied dataset predictions raise `AssertionError` | Medium | No |
| [RS-1](#rs-1-unlisted-features-are-set-to-zero) | Random search | Features missing from the priorities become 0 | **High**: nonsense counterfactuals | Yes |
| [RS-2](#rs-2-model-errors-are-silently-ignored) | Random search | Exceptions from the model are logged and skipped | **High**: a broken model looks like "no answer" | Yes |
| [RS-3](#rs-3-a-forbidden-category-crashes-the-score) | Random search | Scoring a `None` (forbidden) category crashes | Medium | Yes |
| [RS-4](#rs-4-invalid-priorities-are-accepted) | Random search | Invalid priorities (e.g. `min > max`) are accepted | Medium | Yes |
| [PLOT-1](#plot-1-plotting-changes-global-matplotlib-settings) | Plotting | Global Matplotlib settings are changed permanently | Medium | Yes |

Algorithm limitations that are not bugs, but are candidates for improvement,
are listed [at the end](#algorithm-limitations-and-improvement-ideas).

---

## MINLP-1: Allowed ranges are ignored

**Symptom.** `MINLPSearchExplainer` returns counterfactuals whose numerical
features lie outside the `min`/`max` the user allowed:

```python
import numpy as np

from explainit import MINLPSearchExplainer, basic_linear

priorities = {
    "numerical": {
        0: {
            "min": 0.3,
            "max": 0.8,
            "function": lambda x: basic_linear(x, 0.3, 0.8, increasing=False),
        },
        1: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.0, 1.0)},
    },
    "categorical": {},
}
dataset = np.random.default_rng(0).uniform(0, 1, (200, 2))
explainer = MINLPSearchExplainer(
    lambda X: np.asarray(X).sum(axis=1), priorities, np.array([0.6, 0.2]), 1.0, dataset
)
counterfactual = explainer.find_counterfactuals(random_seed=0)
# counterfactual is about [0.006, 0.995]: feature 0 is far below its minimum of 0.3
```

On the benchmark (`benchmarks/`), 0% of MINLP answers respected the allowed
ranges, compared with 100% for random search.

**Root cause.** Stage 0,
`MINLPSearchExplainer._derive_bounds_and_intervals_from_priorities`, replaces
each feature's `min`/`max` with the region where the priority function is
positive *within the dataset's range*. The built-in ramps (`basic_linear`,
`exponential`) keep a positive value (1.0 or the full ramp) outside
`[x0, x1]`, so the derived region is usually the whole dataset range. The
docstring states this override is intentional ("this overrides user min/max"),
but it contradicts the meaning of `min`/`max` in the README and in random
search.

**Suggested fix.** Intersect the derived intervals with the user's
`[min, max]` instead of replacing them.

**Note for experiments.** If published MINLP results were not checked
against the allowed ranges, some of their counterfactuals may lie outside
them.

## MINLP-2: The caller's priorities are modified

**Symptom.** After `find_counterfactuals`, the priorities dictionary passed to
the constructor has new `min`/`max` values and an `allowed_intervals` key.
Reusing the same dictionary for another explainer (as the README examples do)
silently changes that explainer's behaviour.

**Root cause.** `PrioritiesState` stores the caller's dictionary without
copying it, and Stage 0 (`_derive_bounds_and_intervals_from_priorities`)
writes into the per-feature dictionaries.

**Suggested fix.** Deep-copy the priorities in the constructor (functions are
shared by `copy.deepcopy`, which is fine), and keep derived bounds in
separate state.

## MINLP-3: The anchor can carry wrong immutable values

**Symptom.** When the Stage 1 anchor (a dataset row, or a randomly generated
point) is returned as the counterfactual, immutable features (`function:
None`) keep the anchor's values instead of the sample's.

**Root cause.** `_stage1_find_exemplar` → `find_closest_elem` →
`get_rows_in_priorities` skips immutable features when filtering the dataset,
and `find_counterfactuals` returns the anchor unchanged
(`best_feasible_source = "anchor"`).

**Suggested fix.** Overwrite immutable features of the anchor with the
sample's values, or filter dataset rows to those matching them. Re-check the
anchor's prediction afterwards, since the change can move it out of the
target band.

## MINLP-4: Binary search overwrites the explainer's settings

**Symptom.** After `find_counterfactuals_for_binary`, the explainer's
`target` is 0.75 (or 0.25) and its `epsilon` is 0.25, whatever they were
before. A later `find_counterfactuals` call on the same explainer uses these
values.

**Root cause.** `find_counterfactuals_for_binary` assigns `self.target` and
`self.epsilon` to hard-coded values instead of using local variables.

**Suggested fix.** Use local values. More broadly, binary search is a
separate, older copy of the pipeline (see
[limitations](#algorithm-limitations-and-improvement-ideas)); expressing
"probability above the threshold" as a target range would let it reuse the
regression pipeline.

## MINLP-5: Binary results are sorted worst-first

**Symptom.** `find_counterfactuals_for_binary` sorts its results in
*ascending* order of score, and `return_top_n` keeps the first `n`, so it
keeps the least preferable answers.

**Root cause.** The score is `calculate_total_weight`, a benefit where higher
is better, but the sort (`np.argsort(scores)`) and its comment ("lower
cost/weight is better") treat it as a cost.

**Suggested fix.** Sort in descending order.

## MINLP-6: Binary search ignores `expected_counterfactuals`

**Symptom.** `find_counterfactuals_for_binary(expected_counterfactuals=50)`
returns the same results as `expected_counterfactuals=1`: at most one answer
per surviving categorical combination.

**Root cause.** The argument is documented but never used.

**Suggested fix.** Remove the argument, or generate several answers (for
example from different warm starts).

## MINLP-7: Tied predictions crash the anchor search

**Symptom.** If two dataset rows are equally close to the target, the search
fails with `AssertionError: Expected exactly one target exemplar index`. This
is likely with classifiers that output saturated probabilities, or with
duplicated rows.

**Root cause.** `find_closest_elem` asserts that exactly one row has the
minimal distance.

**Suggested fix.** Pick one deterministically (e.g. the first, or the one
closest to the sample).

## RS-1: Unlisted features are set to zero

**Symptom.** In `RandomSearchExplainer`, a feature that does not appear in
`priorities["numerical"]` (or in a categorical group) is set to `0` in every
candidate instead of keeping the person's value. For a sample
`[0.1, 0.1, 5.0]` with priorities only for features 0 and 1, every
counterfactual has `x[2] = 0`.

**Root cause.** `generate_random_samples` and `generate_for_binary` start each
candidate from `np.zeros(n_features)` and only fill the features listed in the
priorities. MINLP starts from the sample instead, so the two methods disagree.

**Suggested fix.** Start each candidate from a copy of the sample.

## RS-2: Model errors are silently ignored

**Symptom.** If the model raises an exception (wrong input shape, missing
columns, a bug), random search logs a warning and continues; the user just
gets an empty result, as if no counterfactual existed.

**Root cause.** A broad `except Exception` around the model call in
`generate_random_samples` and `generate_for_binary`.

**Suggested fix.** Let exceptions propagate.

## RS-3: A forbidden category crashes the score

**Symptom.** `calculate_preference_score` raises `TypeError` for a sample
whose categorical combination has weight `None`, although `None` is the
documented way to forbid a category. `get_preference_breakdown` handles the
same case correctly.

**Root cause.** `float(possible_values.get(current_combo, 0))` with a `None`
value.

**Suggested fix.** Treat `None` as 0, as `get_preference_breakdown` does.

## RS-4: Invalid priorities are accepted

**Symptom.** Priorities with `min > max`, functions returning values outside
`[0, 1]`, or a categorical group whose combinations have the wrong length are
accepted and fail later, or not at all, with confusing results.

**Root cause.** Neither explainer validates the priorities.

**Suggested fix.** Validate priorities once, when the explainer is created,
with clear error messages naming the offending feature.

## PLOT-1: Plotting changes global Matplotlib settings

**Symptom.** Calling any plotting function changes 18 global Matplotlib
settings (fonts, colours, background) for the rest of the user's session, so
the user's own later plots look different.

**Root cause.** `explainit.utils.plot_styles.apply_style` writes to
`plt.rcParams` directly.

**Suggested fix.** Apply the style inside `with plt.rc_context(...)`, or only
to the figures explainit creates.

---

## Algorithm limitations and improvement ideas

These are not bugs (the code does what it is written to do), but they limit
quality or speed. They are the main candidates for the post-refactoring
improvement work.

| Area | Limitation | Idea |
| --- | --- | --- |
| MINLP binary search | A separate, older pipeline: no trust region, residual correction, restoration, or peak lock-in; targets fixed at 0.75 / 0.25. | Treat "class 1" as the target range `(threshold, 1]` and run the regression pipeline. |
| Exact Shapley values | Each feature evaluates all 2^(n-1) coalitions twice: n·2^n model rows in total, of which only 2^n are distinct. | Evaluate the 2^n coalitions once and reuse them for every feature (n-fold fewer model rows). |
| Random search cost | The model is called with one row at a time. | Sample candidates in batches and call the model once per batch. |
| Random search diversity | Several "different" counterfactuals can be identical (e.g. the only allowed category is chosen every time). | De-duplicate, or add a diversity criterion. |
| Random numbers | Seeding uses the legacy `RandomState`. | Use `numpy.random.Generator` and an `rng` argument ([SPEC 7](https://scientific-python.org/specs/spec-0007/)). This changes seeded results. |
| Anchor tolerance | `target_exemplar_epsilon` (default 0.01) is independent of `epsilon`, so a looser `epsilon` does not make dataset anchors easier to find. | Default it to `epsilon`. |
| Reproducibility across machines | MINLP results can differ in the 4th decimal between operating systems and CPUs (SciPy's optimiser and NumPy's CPU-specific code). | Document; consider rounding reported values. |
