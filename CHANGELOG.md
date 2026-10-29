# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
Seeded search results are part of the contract: any change that alters them is
listed explicitly.

## [Unreleased]

### Added

- Top-level imports: `from explainit import RandomSearchExplainer,
  MINLPSearchExplainer, basic_linear, exponential, basic_linear_step,
  plot_priorities`.
- `RandomSearchExplainer.generate_random_samples` accepts
  `n_candidates_per_cf`.
- `explainit.utils.priority_plots.sample_numeric_value` accepts an `rng`.
- Test suite, Ruff configuration, pre-commit hooks, and GitHub Actions CI.
- Correctness tests against exact answers, property-based invariant tests,
  snapshot tests, a register of known bugs, and a quality benchmark with a
  recorded baseline.
- README quick start, `docs/how-it-works.md`, contributing guide, code of
  conduct, security policy, and citation metadata.

### Changed

- `MINLSearchExplainer` is renamed to `MINLPSearchExplainer`. The old name
  still works but emits a `DeprecationWarning`.
- Importing explainit no longer calls `logging.basicConfig(level=DEBUG)`.
  Configure logging in your application to see search progress.
- Seeded searches no longer reset the global NumPy and `random` state. Results
  for a given seed are unchanged.
- `RandomSearchExplainer.display_priorities` and
  `investigate_probability_distribution` no longer write PNG files to
  `./images` by default; they show the plots instead. Pass `save_dir` to write
  files.
- Minimum requirements are now Python 3.12, NumPy 2.0, SciPy 1.13, and
  Matplotlib 3.9 (previously Python 3.13.5; the NumPy/SciPy minimums were not
  declared).
- Packaging metadata moved from `setup.py` to `pyproject.toml`.

### Fixed

- `return_top_n` in `RandomSearchExplainer.generate_random_samples` and
  `generate_for_binary` was accepted but ignored.
- Priority plots with a target distribution used the `boxplot(labels=...)`
  argument removed in Matplotlib 3.11.

### Known issues

Tracked as expected failures in `tests/test_known_bugs.py`:

- `MINLPSearchExplainer` ignores the `min`/`max` of a feature wherever its
  priority function is positive, so answers can leave the allowed range.
- `MINLPSearchExplainer` writes derived bounds into the caller's priorities
  dictionary, changing later runs that reuse it.
- `find_counterfactuals_for_binary` overwrites the explainer's `target` and
  `epsilon`.
- `MINLPSearchExplainer` can return the dataset anchor without resetting
  immutable features to the sample's values.
- `RandomSearchExplainer` sets features missing from the priorities to 0,
  crashes when scoring a forbidden (`None`) category, silently skips model
  errors, and accepts `min > max`.
- Plotting permanently changes global Matplotlib settings.

## [0.1.0]

- Initial version distributed from GitHub.

[Unreleased]: https://github.com/Barszo/explainit_project/commits/main
[0.1.0]: https://github.com/Barszo/explainit_project/commit/6bb46235745ceff8f33b1015f4bb6c1b8b51e1c3
