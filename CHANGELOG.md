# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
Seeded search results are part of the contract: any change that alters them is
listed explicitly.

## [Unreleased]

### Breaking: new API

Explainers are now configured once with the model and the priorities, and
`explain(x, ...)` works for any sample, returning an `Explanation` object.
The search algorithms are unchanged: for the same inputs and seed, results
are identical to 0.1.0.

| 0.1.0 | Now |
| --- | --- |
| `RandomSearchExplainer(model, priorities, sample, target)` | `RandomSearchExplainer(model, priorities, max_iterations=..., candidates_per_counterfactual=..., use_monte_carlo=..., max_tries=...)` |
| `.generate_random_samples(expected_counterfactuals, max_iterations, epsilon, random_seed, ..., return_top_n)` | `.explain(sample, target=..., tolerance=..., n_counterfactuals=..., top_n=..., random_seed=...)` |
| `.generate_for_binary(target_class=..., threshold=...)` | `.explain(sample, target_class=..., threshold=...)` |
| `cfs, preds, scores, iterations = ...` | `result.counterfactuals`, `result.predictions`, `result.scores`, `result.diagnostics["iterations"]` |
| `.calculate_preference_score(x)`, `.get_preference_breakdown(x)` | `.preference_score(x)`, `.preference_breakdown(x)` |
| `.display_priorities(...)`, `.investigate_probability_distribution(...)` | `.plot_priorities(x, ...)`, `.plot_sampling_distributions(...)` |
| `MINLSearchExplainer(model, priorities, sample, target, dataset, target_exemplar_epsilon, epsilon, ...)` | `MINLPSearchExplainer(model, priorities, dataset, settings=MINLPSettings(anchor_tolerance=...))` |
| `.find_counterfactuals(shap_approx, num_samples, max_iterations, trust_region, ...)` | `.explain(sample, target=..., tolerance=..., random_seed=...)` with options in `MINLPSettings(approximate_shapley=..., shapley_samples=..., max_iterations=..., trust_region=..., ...)` |
| `.find_counterfactuals_for_binary(target_class, threshold, max_iterations)` | `.explain(sample, target_class=..., threshold=...)`, `MINLPSettings(binary_max_iterations=...)` |
| `.last_search_result`, `.exemplar_source` | `result.diagnostics` (includes `exemplar_source` and the new `prediction`) |
| `.probe_allowed_region_feasibility(...)` | `.probe_feasibility(sample, target=..., tolerance=...)` |
| `explainit.explainers.minlp_search` | `explainit.explainers.minlp` (one module per algorithm stage) |
| `explainit.utils.priority_plots.sample_numeric_value` | `explainit.sampling.sample_numeric_value` |

Because each `explain` call works on its own copy of the priorities, the
caller's priorities are no longer modified, and binary searches no longer
change later searches (formerly known issues MINLP-2 and MINLP-4).

### Added

- `Explanation` result object with `counterfactuals`, `predictions`,
  `scores`, `valid`, `changes`, `diagnostics` and `to_frame()` (pandas,
  optional: `pip install explainit[pandas]`).
- `MINLPSettings` for the MINLP tuning options.
- Top-level imports: `from explainit import RandomSearchExplainer,
  MINLPSearchExplainer, MINLPSettings, Explanation, basic_linear,
  exponential, basic_linear_step, plot_priorities`.
- `explainit.sampling` with the priority-weighted samplers used by the search
  and the plots.
- Test suite (unit, exact-answer, property-based, snapshot, known-bug and
  documentation tests), a quality benchmark with a recorded baseline, Ruff,
  pre-commit hooks, and GitHub Actions CI.
- README quick start, `docs/how-it-works.md`, `docs/known-issues.md`,
  contributing guide, code of conduct, security policy, and citation
  metadata.

### Changed

- The code is restructured: the MINLP search is a package with one module per
  stage, and random search uses one loop for all target types. Results are
  unchanged.
- Importing explainit no longer calls `logging.basicConfig(level=DEBUG)`.
  Configure logging in your application to see search progress.
- Seeded searches no longer reset the global NumPy and `random` state. Results
  for a given seed are unchanged.
- Plotting methods show plots by default and write files only when
  `save_dir` is given (previously random search wrote to `./images`).
- Minimum requirements are now Python 3.12, NumPy 2.0, SciPy 1.13, and
  Matplotlib 3.9 (previously Python 3.13.5; the NumPy/SciPy minimums were not
  declared).
- Packaging metadata moved from `setup.py` to `pyproject.toml`.

### Removed

- Unused MINLP code: the disconnected "bounds adjustment" suggestions
  (`find_required_bounds_adjustments` and its helpers) and a duplicate
  random-anchor fallback.

### Fixed

- `return_top_n` of the random search was accepted but ignored (now
  `top_n`).
- Priority plots with a target distribution used the `boxplot(labels=...)`
  argument removed in Matplotlib 3.11.

### Known issues

Confirmed bugs and algorithm limitations are documented in
[docs/known-issues.md](docs/known-issues.md). They will be fixed after the
refactoring, which keeps behaviour identical.

## [0.1.0]

- Initial version distributed from GitHub.

[Unreleased]: https://github.com/Barszo/explainit_project/commits/main
[0.1.0]: https://github.com/Barszo/explainit_project/commit/6bb46235745ceff8f33b1015f4bb6c1b8b51e1c3
