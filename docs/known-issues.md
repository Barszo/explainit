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
| [MINLP-3](#minlp-3-the-anchor-can-carry-wrong-immutable-values) | MINLP | The returned anchor can have wrong immutable-feature values | **High**: "unchangeable" features change | Yes |
| [MINLP-5](#minlp-5-binary-results-are-sorted-worst-first) | MINLP binary | Results are sorted worst-first | Low | No (needs two results) |
| [MINLP-7](#minlp-7-tied-predictions-crash-the-anchor-search) | MINLP | Tied dataset predictions raise `AssertionError` | Medium | No |
| [RS-1](#rs-1-unlisted-features-are-set-to-zero) | Random search | Features missing from the priorities become 0 | **High**: nonsense counterfactuals | Yes |
| [RS-2](#rs-2-model-errors-are-silently-ignored) | Random search | Exceptions from the model are logged and skipped | **High**: a broken model looks like "no answer" | Yes |
| [RS-3](#rs-3-a-forbidden-category-crashes-the-score) | Random search | Scoring a `None` (forbidden) category crashes | Medium | Yes |
| [RS-4](#rs-4-invalid-priorities-are-accepted) | Random search | Invalid priorities (e.g. `min > max`) are accepted | Medium | Yes |
| [RS-5](#rs-5-empty-priorities-score-inconsistently) | Random search | Score of empty priorities is 1 in one method, 0 in another | Low | No |
| [PLOT-1](#plot-1-plotting-changes-global-matplotlib-settings) | Plotting | Global Matplotlib settings are changed permanently | Medium | Yes |

Algorithm limitations that are not bugs, but are candidates for improvement,
are listed [at the end](#algorithm-limitations-and-improvement-ideas).

**Resolved by the new API.** Three earlier entries cannot occur with
`explain()`, because every call works on its own copy of the priorities and
takes the target and tolerance as arguments:
MINLP-2 (the caller's priorities dictionary was modified), MINLP-4 (binary
search overwrote the explainer's target and tolerance), and MINLP-6 (binary
search ignored `expected_counterfactuals`; the argument no longer exists).
Results of a single search are unchanged.

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
explainer = MINLPSearchExplainer(lambda X: np.asarray(X).sum(axis=1), priorities, dataset)
result = explainer.explain(np.array([0.6, 0.2]), target=1.0, random_seed=0)
counterfactual = result.counterfactuals[0]
# counterfactual is about [0.006, 0.995]: feature 0 is far below its minimum of 0.3
```

On the benchmark (`benchmarks/`), 0% of MINLP answers respected the allowed
ranges, compared with 100% for random search.

**Root cause.** Stage 0,
`derive_bounds_and_intervals` in `explainit/explainers/minlp/bounds.py`, replaces
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

## MINLP-3: The anchor can carry wrong immutable values

**Symptom.** When the Stage 1 anchor (a dataset row, or a randomly generated
point) is returned as the counterfactual, immutable features (`function:
None`) keep the anchor's values instead of the sample's.

**Root cause.** `filter_rows` in `minlp/anchor.py` skips immutable features
when filtering the dataset, and `MINLPSearch.find_counterfactuals`
(`minlp/search.py`) keeps the anchor unchanged as an incumbent (`Incumbents.set_anchor` in
`minlp/incumbent.py`).

**Suggested fix.** Overwrite immutable features of the anchor with the
sample's values, or filter dataset rows to those matching them. Re-check the
anchor's prediction afterwards, since the change can move it out of the
target band.

## MINLP-5: Binary results are sorted worst-first

**Symptom.** `MINLPSearchExplainer.explain(x, target_class=...)` returns its
counterfactuals in *ascending* order of preference score, so the least
preferable answer comes first.

**Root cause.** The score is the priority benefit, where higher is better,
but `rank` in `minlp/binary.py` sorts with `np.argsort(scores)`, as if it were
a cost.

**Suggested fix.** Sort in descending order.

## MINLP-7: Tied predictions crash the anchor search

**Symptom.** If two dataset rows are equally close to the target, the search
fails with `AssertionError: Expected exactly one target exemplar index`. This
is likely with classifiers that output saturated probabilities, or with
duplicated rows.

**Root cause.** `closest_row` in `minlp/anchor.py` asserts that exactly one
row has the minimal distance.

**Suggested fix.** Pick one deterministically (e.g. the first, or the one
closest to the sample).

## RS-1: Unlisted features are set to zero

**Symptom.** In `RandomSearchExplainer`, a feature that does not appear in
`priorities["numerical"]` (or in a categorical group) is set to `0` in every
candidate instead of keeping the person's value. For a sample
`[0.1, 0.1, 5.0]` with priorities only for features 0 and 1, every
counterfactual has `x[2] = 0`.

**Root cause.** `sample_candidate` in `explainers/random_search.py` starts each
candidate from `np.zeros(n_features)` and only fills the features listed in
the priorities. MINLP starts from the sample instead, so the two methods disagree.

**Suggested fix.** Start each candidate from a copy of the sample.

## RS-2: Model errors are silently ignored

**Symptom.** If the model raises an exception (wrong input shape, missing
columns, a bug), random search logs a warning and continues; the user just
gets an empty result, as if no counterfactual existed.

**Root cause.** A broad `except Exception` around the model call in
`RandomSearchExplainer.explain`.

**Suggested fix.** Let exceptions propagate.

## RS-3: A forbidden category crashes the score

**Symptom.** `RandomSearchExplainer.preference_score` raises `TypeError` for a
sample whose categorical combination has weight `None`, although `None` is the
documented way to forbid a category. `preference_breakdown` handles the same
case correctly.

**Root cause.** `float(weights.get(combination, 0))` in `preference_score`
(`explainers/random_search.py`) with a `None` value.

**Suggested fix.** Treat `None` as 0, as `preference_breakdown` does.

## RS-4: Invalid priorities are accepted

**Symptom.** Priorities with `min > max`, functions returning values outside
`[0, 1]`, or a categorical group whose combinations have the wrong length are
accepted and fail later, or not at all, with confusing results.

**Root cause.** Neither explainer validates the priorities.

**Suggested fix.** Validate priorities once, when the explainer is created,
with clear error messages naming the offending feature.

## RS-5: Empty priorities score inconsistently

**Symptom.** With no scored features, `preference_score` returns `1.0` while
`preference_breakdown` reports an overall score of `0.0`.

**Root cause.** Different fallbacks in the two functions in
`explainers/random_search.py`.

**Suggested fix.** Return `0.0` from both (the sum of no priorities).

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
