# Continuous-target MINLP experiment

This experiment runs `MINLSearchExplainer.find_counterfactuals` against
regression models (continuous target). It is split into small,
single-purpose stages so a dataset, model, or priority configuration can
be iterated on without touching the rest of the pipeline.

The `random_runner.py` script reproduces the same configuration with
`RandomSearchExplainer` so MINLP results can be compared against a
random-search baseline on identical priorities.

There are now **three branches** after a model is trained (stage 2):

* the **priority / MINLP branch** (stages 3-7 below), which is where our own
  `MINLSearchExplainer` runs against declarative priority sets and writes one
  JSON result per `(sample, target)` pair,
* the **priority-methods branch** under `priority_methods/`, a self-contained
  package that mirrors the `standard_methods/` structure (selection → methods
  → runner → config → results explorer) but runs the **priority** methods
  (MINLP + a random-search baseline) against the declarative priority sets and
  persists per-CF metrics **including a `priority_score`**. Documented in
  ["Priority-methods branch"](#priority-methods-branch-packaged-minlp--baseline)
  below, and
* the **standard-methods baseline branch** under `standard_methods/`, which
  runs well-known regression counterfactual algorithms (DiCE, Wachter, sparse
  Wachter, prototype-guided, growing spheres, Nelder-Mead, Bayesian
  optimisation, random search) against a *predicted-target* dataset and
  persists comparable metrics. This branch is documented in
  ["Standard-methods baseline stage"](#standard-methods-baseline-stage-predicted-target-dataset)
  right after stage 2 and is independent of the priority sets.

## Directory layout

```
explainit/experiments/continuous_minlp/
├── README.md                     <- this file
├── __init__.py
├── data_setup.py                 <- stage 1: dataset pickles
├── model_setup.py                <- stage 2: trained Keras models
├── priority_sets.py              <- stage 3: declarative priority sets (edit me)
├── priorities_selection.py       <- stage 4: workbench for analyser plots
├── priorities_explorer.ipynb     <- notebook: inspect a sample's priorities
├── minlp_test_config.yaml        <- stage 5: experiment configuration
├── minlp_runner.py               <- stage 6: MINLP search runner
├── random_runner.py              <- stage 7: random-search baseline runner
│
├── priority_methods/             <- packaged priority branch (runs after stage 2)
│   ├── selection.py                     <- sample/target + priority context
│   ├── methods.py                       <- MINLP + random-search + priority_score
│   ├── config.yaml                      <- priority-methods configuration
│   ├── runner.py                        <- runs the methods + writes results
│   ├── results_explorer.ipynb           <- notebook: explore the run results
│   └── results/<dataset_key>/{samples.csv, counterfactuals.csv,
│                              metrics_summary.csv, summary.json,
│                              run_config.json}
│
├── standard_methods/             <- baseline branch (runs after stage 2)
│   ├── predicted_dataset_setup.py       <- stage 2b: predicted-target dataset
│   ├── predicted_dataset_explorer.ipynb <- notebook: inspect predicted dataset
│   ├── selection.py                     <- sample/target + actionability logic
│   ├── methods.py                       <- regression CF method registry
│   ├── config.yaml                      <- standard-methods configuration
│   ├── runner.py                        <- runs the methods + writes results
│   ├── results_explorer.ipynb           <- notebook: explore the run results
│   ├── predicted_data/<dataset_key>/data.pkl
│   └── results/<dataset_key>/{samples.csv, counterfactuals.csv,
│                              metrics_summary.csv, summary.json,
│                              run_config.json}
│
├── data/<dataset_key>/data.pkl
├── data_analysis/<dataset_key>/{numerical_features.csv, categorical_features.csv}
├── models/<dataset_key>/model.keras
├── analysis/<dataset_key>/<sample_idx>_<target>/{coverage.txt, dataset/, priorities/, analysis_summary.txt}
└── results/<dataset_key>/<sample_idx>_<target>/{minlp.json, random.json}
```

## End-to-end flow

### 1. Prepare datasets

```bash
python -m explainit.experiments.continuous_minlp.data_setup
# or just one:
python -m explainit.experiments.continuous_minlp.data_setup --datasets diabetes
# overwrite existing pickles:
python -m explainit.experiments.continuous_minlp.data_setup --force
```

If you prefer running by file path, this also works from the repo root:

```bash
python explainit/experiments/continuous_minlp/data_setup.py --force
```

Each dataset gets:

* `X_train` / `X_test` where **numerical** columns are standard-scaled and
  **categorical** columns are one-hot encoded (one column per category,
  e.g. `sex` becomes `sex=0` / `sex=1`),
* a MinMax-scaled `y_train` / `y_test` (target lives in `[0, 1]`),
* the fitted `x_scaler` (numerical only) and `y_scaler` to invert
  predictions later,
* `numerical_features` and `categorical_groups` metadata (column indices,
  category codes, source values) used by the priority builder,
* automatic feature-type analysis CSVs under
  `data_analysis/<dataset_key>/`,
* persisted to `data/<dataset_key>/data.pkl`.

Add a new dataset by registering a loader in `DATASETS` inside
`data_setup.py`.

#### Automatic categorical detection + manual overrides

`data_setup.py` now infers categorical features automatically and logs
for each feature why it was classified as categorical or numerical.
Heuristics:

* non-numeric dtype -> categorical,
* boolean dtype -> categorical,
* numeric + integer-like + low cardinality (`<=20` unique values and
  unique/non-null ratio `<=5%`) -> categorical.

After automatic detection, `CATEGORICAL_FEATURES` in `data_setup.py`
still works as a manual override and is applied on top:

```python
CATEGORICAL_FEATURES = {
    "diabetes": ["sex"],   # force these columns to categorical
}
```

During setup, two CSV files are written to
`data_analysis/<dataset_key>/`:

* `numerical_features.csv` with: feature name, whether auto-detected as
  categorical, unique/all/non-null counts, non-null %, min, max, mean,
  median, std.
* `categorical_features.csv` with: feature name, unique/all/non-null
  counts, non-null %, and top-20 most frequent values with percentages.

At the end of preprocessing, the terminal log explicitly asks you to
review these CSVs and update `CATEGORICAL_FEATURES` if you want to
override the automatic decisions.

After changing this registry (or any priority set that touches
categoricals) you **must regenerate** the data and model, because the
feature layout changes:

```bash
python -m explainit.experiments.continuous_minlp.data_setup  --datasets diabetes --force
python -m explainit.experiments.continuous_minlp.model_setup --datasets diabetes --force
```

### 2. Train models

```bash
python -m explainit.experiments.continuous_minlp.model_setup
python -m explainit.experiments.continuous_minlp.model_setup --datasets diabetes --epochs 80
python -m explainit.experiments.continuous_minlp.model_setup --force
```

Models land at `models/<dataset_key>/model.keras`. The default
architecture is a two-layer MLP; add a custom builder by registering it
in `MODEL_BUILDERS` inside `model_setup.py`.

The measures of each model are saved in `model_analysis/model_analysis.csv`

## Standard-methods baseline stage (predicted-target dataset)

Everything under `standard_methods/` is a **self-contained baseline branch**
that runs *after stage 2* and is completely independent of the priority sets
and MINLP flow (stages 3-7). It targets the **model's own predictions** (not
the ground-truth labels) and runs a battery of well-known regression
counterfactual (CF) methods so their metrics can be compared later against
`MINLSearchExplainer`.

The full flow is: **2b build predicted-target dataset → explore it →
configure → run methods → explore results**.

### 2b. Build the predicted-target dataset

Counterfactual search targets the *model*, so this stage loads the stage-1
dataset and stage-2 model, predicts the (scaled) target for every row, and
writes a new dataset pickle where `y_train` / `y_test` are the **model
predictions**. Everything else (feature matrix, scalers, categorical
metadata) is carried over unchanged, and the original labels are kept under
`y_train_true` / `y_test_true`.

```bash
python -m explainit.experiments.continuous_minlp.standard_methods.predicted_dataset_setup
# or one dataset / overwrite:
python -m explainit.experiments.continuous_minlp.standard_methods.predicted_dataset_setup --datasets diabetes --force
```

Output: `standard_methods/predicted_data/<dataset_key>/data.pkl`.

Re-run this with `--force` whenever you retrain the model, so the predicted
target reflects the current model.

### 2c. Explore the predicted-target dataset

`predicted_dataset_explorer.ipynb` loads the predicted dataset and plots the
predicted-target distribution (train/test), predicted-vs-true, and each
feature against the predicted target. Set `DATASET = "diabetes"` in the first
cell and run all cells. Use it to sanity-check the target range before
choosing sample/target offsets in the config.

### 2d. Configure the run (`standard_methods/config.yaml`)

Targets are expressed in the MinMax-scaled `[0, 1]` space. For each selected
sample the desired target is `model_prediction + target_offset`; samples
whose resulting target falls below `skip_if_target_below` are skipped.

```yaml
defaults:
  epsilon: 0.05
  n_cfs: 5                  # default number of CFs per (sample, method)

experiments:
  - dataset: diabetes

    selection:
      strategy: indices         # "indices" or "random"
      sample_indices: [60, 2, 52, 18, 27, 26, 38, 50, 24, 33]
      # n_samples: 10           # only for random strategy
      # seed: 42                # only for random strategy
      target_offset: -0.3       # target = prediction - 0.3 (scaled)
      skip_if_target_below: 0.0 # drop samples whose target would be < 0

    actionability:
      immutable: [sex]          # logical feature name -> all one-hot cols pinned
      bounds:
        age: {direction: increasing}   # age may only rise, capped at dataset max
      # every other feature is free to change without bounds

    methods:
      - name: dice
        n_cfs: 5              # per-method override of defaults.n_cfs
        params: {total_cfs: 5, method: genetic, backend: sklearn}
      - name: wachter
        n_cfs: 5
        params: {learning_rate: 0.1, max_iterations: 1000, proximity_weight: 0.005, seed: 42}
      - name: random_search
        n_cfs: 5
        params: {max_iterations: 3000, seed: 42}
      # ... other methods ...
```

Counterfactual count:

* `defaults.n_cfs` sets how many counterfactuals each method returns per
  sample; a per-method `n_cfs` overrides it for that method.
* A method that declares it cannot return more than one distinct CF
  (`supports_multiple = False` in `methods.py`) is clamped to a single CF
  regardless of `n_cfs`. All methods currently shipped support multiple CFs.

Selection keys:

* `strategy: indices`
  * requires `sample_indices: [...]`.
  * uses the provided test-set indices in the listed order.
* `strategy: random`
  * uses `n_samples` and `seed`.
  * draws a random permutation of the test set and keeps the first valid rows.
* `target_offset`, `skip_if_target_below` (and optional `skip_if_target_above`)
  are applied in both strategies.

Random-strategy example:

```yaml
selection:
  strategy: random
  n_samples: 10
  seed: 42
  target_offset: -0.3
  skip_if_target_below: 0.0
```

Actionability keys:

* `immutable`: logical feature names that must not change. A categorical
  feature (e.g. `sex`) pins **all** of its one-hot columns.
* `bounds`: per logical feature, one of
  * `{direction: increasing}` → `lo = sample value`, `hi = dataset max`
    (feature may only increase),
  * `{direction: decreasing}` → `lo = dataset min`, `hi = sample value`,
  * `{min: <v>, max: <v>}` → an explicit box.
* Any feature not listed under `immutable`/`bounds` is free to move (no
  bounds; samplers fall back to the dataset column range).

### 2e. Run the standard methods

```bash
python -m explainit.experiments.continuous_minlp.standard_methods.runner
python -m explainit.experiments.continuous_minlp.standard_methods.runner --dataset diabetes
python -m explainit.experiments.continuous_minlp.standard_methods.runner --config standard_methods/config.yaml
```

For each `(sample, method)` pair the runner generates up to `n_cfs`
counterfactuals, records each one (indexed by `cf_index`), checks validity
(`|prediction - target| <= epsilon`), and computes per-CF proximity /
sparsity / timing metrics.

#### Available methods

Referenced by `name` in the config (`methods.py` registry):

| name | description |
| --- | --- |
| `dice` | Official `dice-ml`, regression mode (`desired_range`), model-agnostic search (default `genetic`). |
| `wachter` | Gradient descent: prediction MSE + L2 proximity. |
| `sparse_wachter` | Wachter with an added L1 (elastic-net) term for sparser edits. |
| `prototype` | Gradient descent pulled toward training points whose prediction is near the target. |
| `growing_spheres` | Model-agnostic: expanding L2 shells around the sample. |
| `nelder_mead` | Gradient-free simplex search (scipy). |
| `bayesian_optimization` | scikit-learn Gaussian-process surrogate with Expected-Improvement acquisition. |
| `random_search` | Uniform random sampling within the box; keeps the closest valid CF. |

All methods honour `immutable` features and the per-feature bounds. Gradient
methods need the Keras model; the model-agnostic ones only call `predict`.

### Outputs

Written to `standard_methods/results/<dataset_key>/`:

* `samples.csv` — one row per selected sample: `sample_id`,
  `original_prediction`, `target`, and every feature value. This is the
  **link key** for the other tables.
* `counterfactuals.csv` — **one row per counterfactual**, i.e. per
  `(sample_id, method, cf_index)`. Columns: `sample_id`, `method`,
  `cf_index`, `target`, `original_prediction`, `cf_prediction`, `validity`,
  `abs_pred_error`, `l1`, `l2`, `n_changed`, `sparsity_fraction`,
  `iterations`, `time_seconds`, `error`, and one `cf__<feature>` column per
  feature holding the CF's value. `cf_index` runs `0..n_cfs-1` for the CFs a
  method returned for that sample.
* `metrics_summary.csv` — one row per method with run-level aggregates:
  `dataset`, `method`, `n_cfs_requested`, `n_samples`,
  `n_samples_with_valid`, `sample_validity_rate` (fraction of samples with at
  least one valid CF), `n_cfs_total`, `n_cfs_valid`, `cf_validity_rate`
  (fraction of returned CFs that are valid), `avg_abs_pred_error`, `avg_l1`,
  `avg_l2`, `avg_n_changed`, `avg_sparsity_fraction`, `avg_iterations`,
  `avg_time_seconds`. The `avg_*` values are computed over the *valid* CFs
  only.
* `summary.json` — the same per-method summary in machine-readable form
  (`{"dataset": ..., "methods": [ ... ]}`).
* `run_config.json` — a snapshot of everything needed to interpret the run:
  `epsilon`, `n_samples_selected`, the resolved `selection` and
  `actionability` config, the `methods` list (each with its resolved `n_cfs`
  and params), and the feature layout (`feature_names`, `numerical_features`,
  `categorical_groups`) plus a UTC timestamp.

Join `counterfactuals.csv` to `samples.csv` on `sample_id` to compare each CF
against its original instance.

> Metrics are computed in the scaled feature space and are intentionally a
> superset of the MINLP result metrics, so the two branches can be compared
> later. New metrics can be appended without breaking the existing columns.

### 2f. Explore the results (`results_explorer.ipynb`)

`results_explorer.ipynb` is an interactive notebook for reading a completed
run out of `results/<dataset_key>/`. Run all cells and drive it with the
widgets. It has three parts.

**Part 1 - Selection.** Pick a results dataset from the dropdown and click
*Load*; the initial samples used for the run (`samples.csv`) are displayed.
Then choose one CF method and one sample and click *Use selection* to fix the
focus for Part 2.

**Part 2 - Sample analysis** (all plots use the *selected method* and
*selected sample*):

* **Counterfactual table** — the original sample followed by one row per
  counterfactual (`cf_index`) that the selected method produced for it.
* **1D distribution** — pick any feature or the target. Continuous features
  render as a histogram of the dataset; categorical features as category
  bars. The sample value and the CF value(s) are overlaid; an *all methods*
  toggle overlays every method's CF instead of just the selected one. When
  the target axis is shown, its valid band is highlighted.
* **2D distribution** — plot two features against each other over the dataset
  background. Two continuous features use a 2D histogram; if either axis is
  categorical the dataset is shown as a jittered scatter with labelled
  category ticks. The sample is a black X connected to each CF; every CF gets
  a distinct colour from a continuous `turbo` colormap (spread across all
  plotted CFs) with its `cf_index` printed inside the dot. When the target is
  on an axis its valid band is shaded green.
* **CF predictions vs target** — one bar per counterfactual showing its
  predicted value against the desired target (green line) and the valid band
  (shaded), with the original prediction as a dashed reference. Bars are
  green when the CF is valid, grey otherwise. A companion table lists each
  CF's distance from the target, sorted by `abs_pred_error`.

**Part 3 - Method metrics** (compare every method across the whole run):

* **Per-method averages** — a grid of bar charts from `metrics_summary.csv`:
  sample validity rate, CF validity rate, number of valid CFs, average L1 and
  L2 (proximity), average #changed and sparsity fraction (sparsity), average
  `|pred - target|`, average time, and average iterations. Averages are over
  valid CFs.
* **Per-CF distributions** — box plots from `counterfactuals.csv` showing the
  spread across individual CFs for L1, L2, sparsity fraction, #changed,
  `|pred - target|`, and time.

> The baseline methods do not compute a preference/priority score (that
> belongs to the MINLP branch), so no priority metric is shown here.

## Priority-methods branch (packaged MINLP + baseline)

Everything under `priority_methods/` is a **self-contained package** that runs
*after stage 2* and mirrors the `standard_methods/` layout, but drives the
**priority** methods against the declarative priority sets from
`priority_sets.py`. It runs `MINLSearchExplainer` (MINLP) and a random-search
baseline on **identical priorities**, and persists the same result tables as
`standard_methods/` **plus a per-CF `priority_score`** so the branches are
directly comparable.

Targets live in the MinMax-scaled `[0, 1]` space: for each selected sample the
target is `model_prediction + target_offset`.

The flow is: **explore priorities → configure → run methods → explore
results**.

### P1. Explore a sample's priorities (`priorities_explorer.ipynb`)

`priorities_explorer.ipynb` is a self-contained notebook to inspect what a
priority set implies for one sample **before** running the methods. Pick a
dataset, a priority set (`set1` / `set2`), and a sample index; it plots each
feature's materialised priority function (with the sample value marked) and
breaks down the sample's own priority score. Use it to sanity-check that the
peaks/decays match your intent and that features stay positive across the
range (so MINLP has a feasible region).

### P2. Configure the run (`priority_methods/config.yaml`)

```yaml
defaults:
  epsilon: 0.05
  n_cfs: 5                  # default CFs per method (MINLP is clamped to 1)

experiments:
  - dataset: diabetes
    priority_set: set1       # which set from priority_sets.py

    selection:
      strategy: indices      # "indices" or "random"
      sample_indices: [60, 2, 52, 18, 27, 26, 38, 50, 24, 33]
      # n_samples: 10        # only for random strategy
      # seed: 42             # only for random strategy
      target_offset: -0.3    # target = prediction - 0.3 (scaled)
      skip_if_target_below: 0.0

    methods:
      - name: minlp
        n_cfs: 1             # MINLP is single-shot
        params:
          shap_approx: true
          shap_num_samples: 200
          max_iterations: 10
          patience: 5
          target_exemplar_epsilon: 0.10
      - name: random_search
        n_cfs: 5
        params: {max_iterations: 10000, use_monte_carlo: true, seed: 42}
```

Selection keys behave exactly like the standard-methods branch (`indices` vs
`random`, `target_offset`, `skip_if_target_below/above`). The priorities
(including which features are actionable and their search bounds) come from the
chosen `priority_set`, not from an `actionability` block.

### P3. Run the methods

```bash
python -m explainit.experiments.continuous_minlp.priority_methods.runner
python -m explainit.experiments.continuous_minlp.priority_methods.runner --dataset diabetes
```

For each `(sample, method)` pair the runner builds the sample-specific
priorities, generates up to `n_cfs` counterfactuals, checks validity
(`|prediction - target| <= epsilon`), and records per-CF proximity / sparsity
/ timing metrics **and the `priority_score`** (sum of the per-feature priority
weights at the CF's values). MINLP failures (e.g. an infeasible exemplar
search under very strict priorities) are caught per sample and recorded as
zero counterfactuals rather than aborting the run.

### Outputs

Written to `priority_methods/results/<dataset_key>/`, identical in shape to the
standard-methods outputs with two additions:

* `counterfactuals.csv` — adds a **`priority_score`** column (per CF).
* `metrics_summary.csv` — adds **`avg_priority_score`** (over valid CFs).
* `samples.csv`, `summary.json`, `run_config.json` — as in the standard branch
  (`run_config.json` also records the `priority_set`).

### P4. Explore the results (`priority_methods/results_explorer.ipynb`)

Mirrors `standard_methods/results_explorer.ipynb` (same three parts and
widgets), reading from `priority_methods/results/` and using the model's
predictions on the training set as the target-axis background. The
`priority_score` is surfaced throughout: in the Part 2 counterfactual table, as
a dedicated bar chart next to "CF predictions vs target", and in the Part 3
per-method (`avg_priority_score`) and per-CF (`priority_score`) metric panels.

### 3. Author priorities

`priority_sets.py` is where you declare **how desirable** each feature
value is when searching for a counterfactual. You do **not** write any
search logic -- you only describe, per feature, either a *priority
function* (numerical features) or a *weights mapping* (categorical
features).

#### The structure

`PRIORITY_SETS` is a nested dictionary:

```
dataset name  ->  set name  ->  "numerical"   ->  feature name  ->  function
                                "categorical" ->  feature name  ->  {code: weight}
```

Features are keyed by their **logical** name (`sex`), not by the one-hot
column names (`sex=0`, `sex=1`) -- the builder expands them for you.

For example:

```python
PRIORITY_SETS = {
    "diabetes": {                 # dataset key (matches data_setup.py)
        "default": {              # set name (referenced by the config/CLI)
            "numerical": {
                "bmi": linear_priority(x0=2.0, x1=-2.0, increasing=False),
                "age": constant_priority(0.5),
                # ... one entry per remaining numerical feature ...
            },
            "categorical": {
                "sex": {0: 1.0, 1: 0.5},   # code 0 preferred over code 1
            },
        },
    },
}
```

To add another configuration for the same dataset, add a sibling set:

```python
"diabetes": {
    "default": { ... },
    "aggressive": { "numerical": { ... }, "categorical": { ... } },
}
```

#### Numerical features: a function `value -> [0, 1]`

A numerical priority is any function that takes a feature value and
returns a preference in `[0, 1]` (`1.0` = most preferred, `0.0` =
unacceptable). Ready-made helpers cover the common shapes:

```python
from explainit.experiments.continuous_minlp.priority_sets import (
    linear_priority, exponential_priority, constant_priority,
    interval_priority, numerical_entry, NON_ACTIONABLE,
)

# Ramp: 0 below x0, rising to 1 at x1 (then stays 1).
linear_priority(x0=10, x1=50, increasing=True)

# Decreasing ramp: 1 below x0, falling to 0 at x1.
linear_priority(x0=10, x1=50, increasing=False)

# Same idea but with a steeper exponential transition.
exponential_priority(x0=10, x1=50, increasing=True, a=5.0)

# Flat preference everywhere (no opinion, free to move).
constant_priority(0.5)

# Box: weight inside [low, high], 0 outside ("only this band is acceptable").
interval_priority(low=10, high=50, weight=1.0)
```

You can also write your own. Anything callable works, e.g. a lambda that
"increases from 0 at 10 to 1 at 50 and is 0 elsewhere":

```python
"some_feature": lambda v: 1.0 if 10 <= v <= 50 else 0.0,
```

> Note: diabetes features are standard-scaled, so their values sit roughly
> in `[-3, 3]` rather than raw units like "age in years". Look at
> `coverage.txt` / the dataset plots (stage 4) to choose sensible numbers.

#### Sample- and dataset-relative priorities

The helpers above return **static** functions that know nothing about the
sample being explained. Many realistic preferences are *relative* -- e.g.
"prefer values just below the sample's current value" or "decay toward the
dataset maximum". Those are expressed with an **anchor** (a point resolved
per sample/feature at build time) plus `peak_priority`, which builds a
single-peak function that decays to each side.

```python
from explainit.experiments.continuous_minlp.priority_sets import (
    peak_priority, at_sample, at_min, at_max, at_value,
)

# Peak (height 1.0) just above the sample, decaying linearly to the dataset
# min on the left and exponentially to the dataset max on the right.
peak_priority(
    peak_at=at_sample(offset=0.5), peak_value=1.0,
    left=at_min(),  left_shape="linear",
    right=at_max(), right_shape="exponential",
)
```

Anchors (all take an `offset` and/or `pct`):

* `at_sample(offset=0.0, pct=0.0)` — the sample's own value for this feature.
* `at_min(...)` / `at_max(...)` — the dataset column min / max.
* `at_value(v)` — a literal (scaled) value.

Units: features are scaled, so an anchor `offset` is an **absolute** shift in
that scaled space, while `pct` is a **fraction of the feature's dataset
range** (`pct=-0.20` == "20% of the range below"; because scaling is linear
this matches 20% of the raw range too).

`peak_priority` parameters:

* `peak_at` — anchor of the peak; `peak_value` — its height in `[0, 1]`.
* `left` / `right` — anchors where each side reaches 0. Pass `None` for a
  **hard cutoff** (0 beyond the peak on that side).
* `left_shape` / `right_shape` — `"linear"` or `"exponential"` (`a` controls
  exponential steepness).

> Tip: hard cutoffs (`left=None`, `right=None`, or a `right=at_sample()` that
> forbids everything above the sample) can make the priority-filtered dataset
> collapse to a handful of rows, which starves the MINLP exemplar search. If
> MINLP reports "no elements fulfilling the requirements" or "no
> positive-priority region", relax the cutoffs to soft decays toward
> `at_min()` / `at_max()` so every feature stays positive across the range.

For full control you can also build your own `ContextualPriority` (a factory
`build(FeatureContext) -> f(value)`); `peak_priority` is the common case.

The shipped diabetes `set1` / `set2` use these helpers: both share every
non-serum priority (via a `_shared_numerical()` helper) and pin `sex` as
non-actionable, and differ **only** in the five serum features (`s1`..`s5`),
whose peak sits 20% of the range below the sample (height 0.7) in `set1`
versus 40% below (height 0.5) in `set2`.

#### Search bounds (min / max)

Each numerical feature is searched within `[min, max]`. By default these
come from the dataset column min/max, so you usually do **not** set them.
To override, wrap the function with `numerical_entry`:

```python
"bmi": numerical_entry(
    linear_priority(x0=2.0, x1=-2.0, increasing=False),
    min_val=-1.0,            # optional lower bound
    max_val=1.5,             # optional upper bound
    use_dataset_bounds=True, # default
),
```

* With `use_dataset_bounds=True` (default), the dataset min/max take
  priority on conflict: the final bounds are the *intersection* of your
  bounds and the dataset range (e.g. if the dataset min is higher than
  your `min_val`, the dataset min wins).
* Set `use_dataset_bounds=False` to use your `min_val`/`max_val` verbatim.
* Independently of bounds, **a value where the function returns 0 is
  rejected** -- so `interval_priority` / a function that drops to 0 is the
  way to carve out forbidden ranges inside the bounds.

#### Categorical features: a `{code: weight}` mapping

A categorical feature must be classified as categorical by
`data_setup.py` (automatic detection, optionally forced with
`CATEGORICAL_FEATURES`). In the priority set, list **every** category
code and its relative weight (`0` = forbidden, higher = more preferred):

```python
"categorical": {
    "profession": {0: 0.0, 1: 0.5, 2: 1.0},   # unemployed / teacher / accountant
},
```

The codes are the original (pre-one-hot) integer values. The builder
expands them into one-hot states and the search keeps **exactly one**
category active at a time (it never produces e.g. `sex=0` and `sex=1`
both set). The diabetes `default` set uses this for `sex`:

```python
"categorical": {
    "sex": {0: 1.0, 1: 0.5},   # both allowed; code 0 preferred
},
```

You must list all codes; a missing or unknown code raises a clear error.
To forbid a category entirely, give it weight `0`.

#### Non-actionable features (leave a feature unchanged)

**Every feature in the dataset must appear exactly once** in the set
(under `numerical` or `categorical`). If one is missing you get a clear
error naming it. To declare that a feature must keep its original value
(the search may not change it), set it to `NON_ACTIONABLE`:

```python
"numerical": {
    "age": NON_ACTIONABLE,   # fixed at the sample's value
    ...
}
```

(Equivalently, `numerical_entry(None)` marks it non-actionable too.) For a
categorical feature, the equivalent of "leave it unchanged" is to give
every other code weight `0`, or simply rely on the search keeping the
sample's category when it has no incentive to switch.

### 4. Inspect priorities + dataset

After editing `priority_sets.py`, run `priorities_selection.py` to see
what your new priorities imply before running expensive MINLP search.

You can configure this script in two ways:

* **Option 1 (edit defaults in file):** open `priorities_selection.py`
  and change:

```python
USER_DATASET = "diabetes"
USER_PRIORITY_SET = "default"
USER_SAMPLE_INDICES = (0, 5)
USER_TARGETS_SCALED = (0.25, 0.75)
```

* **Option 2 (leave file unchanged, override from CLI):**

```bash
python -m explainit.experiments.continuous_minlp.priorities_selection \
    --dataset diabetes --priority-set default \
    --sample 0 --sample 5 --target 0.25 --target 0.75
```

You can combine both (set general defaults in file, override specific
samples/targets from CLI while experimenting).

#### Practical run examples

Run one `(sample, target)` pair:

```bash
python -m explainit.experiments.continuous_minlp.priorities_selection \
    --dataset diabetes --priority-set linear_v1 \
    --sample 3 --target 0.6
```

Compare two samples at one target:

```bash
python -m explainit.experiments.continuous_minlp.priorities_selection \
    --dataset diabetes --priority-set linear_v1 \
    --sample 3 --sample 15 --target 0.6
```

Compare one sample against two targets:

```bash
python -m explainit.experiments.continuous_minlp.priorities_selection \
    --dataset diabetes --priority-set linear_v1 \
    --sample 3 --target 0.3 --target 0.8
```

Outputs land under
`analysis/<dataset_key>/<sample_idx>_<target>/`:

* `coverage.txt` — the same per-feature bounds table the MINLP workflow
  log emits (bounds, dataset min/max, allowed space %, allowed points %).
* `dataset/` — feature/target distribution plots (delegated to
  `explainit.utils.dataset_analyzer.analyze_dataset`).
* `priorities/` — priority surface plots (delegated to
  `explainit.utils.priority_plots.plot_priorities`).
* `analysis_summary.txt` — coverage + closest-exemplar summary.

When multiple `(sample, target)` pairs are configured, an extra folder
`analysis/<dataset_key>/combined/` shows all of them overlaid on the same
priority plots (samples and exemplars share a colour and are linked by a
translucent connector).

How to interpret outputs quickly:

* `coverage.txt`: check if bounds are too tight/too loose (look at
  allowed space % and allowed points %).
* `priorities/` plots: verify the preference shape matches your intent
  (monotonic up/down, flat, steepness).
* `analysis_summary.txt`: inspect which exemplar was chosen for each
  target and whether it makes sense.

Recommended iteration loop:

1. Edit the priority set in `priority_sets.py`.
2. Run `priorities_selection.py` for a small set of pairs.
3. Check `coverage.txt` + plots.
4. Repeat until priorities look right.
5. Only then run `minlp_runner.py`.

### 5. Configure the experiment

Edit `minlp_test_config.yaml`. The top-level structure is:

```yaml
defaults:
  epsilon: 0.05
  ...

experiments:
  - dataset: diabetes
    priority_set: default
    samples: [0, 5, 12]
    targets: [0.25, 0.75]
    n_counterfactuals: 1
```

Each experiment runs every cartesian `(sample, target)` pair. Per-entry
keys override the `defaults` block.

### 6. Run MINLP search

```bash
python -m explainit.experiments.continuous_minlp.minlp_runner
python -m explainit.experiments.continuous_minlp.minlp_runner --config minlp_test_config.yaml
python -m explainit.experiments.continuous_minlp.minlp_runner --dataset diabetes
```

Each pair produces `results/<dataset_key>/<sample_idx>_<target>/minlp.json`
with the headline metrics (validity, iterations, time) plus the resulting
counterfactual vector and the full per-iteration history.

### 7. Run random-search baseline

```bash
python -m explainit.experiments.continuous_minlp.random_runner
```

Writes `results/<dataset_key>/<sample_idx>_<target>/random.json` next to
the MINLP result, so the two files can be diffed directly.

## Persisted metrics (today)

Each result JSON includes (at minimum):

* `dataset`, `priority_set`, `sample_index`, `target_scaled`, `epsilon`
* `original_prediction`, `counterfactual`, `counterfactual_prediction`
* `validity` — `True` when the counterfactual lies within `epsilon` of
  the target
* `iterations` — refinement iterations (MINLP) or iteration index when
  the best CF was found (random search)
* `time_seconds`
* MINLP-only: `reached_target`, `stop_reason`, `history`
* Random-only: `iterations_per_cf`, `n_counterfactuals_found`,
  `preference_score`

More metrics can be added later without breaking the existing JSON keys.

## Adding a new dataset (recipe)

1. Add a raw loader to `data_setup.py` and register it in `DATASETS`
   (and `TARGET_NAMES` if you want a friendly target label).
2. Run `data_setup.py` and inspect
   `data_analysis/<dataset_key>/{numerical_features.csv,categorical_features.csv}`.
   Then add manual overrides in `CATEGORICAL_FEATURES` only where you
   want to force a different categorical decision.
3. Optionally register a custom model builder in `model_setup.py`
   (`MODEL_BUILDERS`); otherwise the default MLP is used.
4. Add a priority set under `PRIORITY_SETS[<dataset_key>]["default"]` in
   `priority_sets.py` (one entry per logical feature: a
   function/`NON_ACTIONABLE` under `numerical`, or a `{code: weight}`
   mapping under `categorical`).
5. Add an entry to `experiments:` in `minlp_test_config.yaml`.
6. `data_setup` → `model_setup` → `priorities_selection` →
   `minlp_runner` → `random_runner`. Re-run `data_setup --force` and
   `model_setup --force` whenever you change `CATEGORICAL_FEATURES` (the
   feature layout changes).
