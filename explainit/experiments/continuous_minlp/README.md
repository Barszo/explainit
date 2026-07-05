# Continuous-target MINLP experiment

This experiment runs `MINLSearchExplainer.find_counterfactuals` against
regression models (continuous target). It is split into small,
single-purpose stages so a dataset, model, or priority configuration can
be iterated on without touching the rest of the pipeline.

The `random_runner.py` script reproduces the same configuration with
`RandomSearchExplainer` so MINLP results can be compared against a
random-search baseline on identical priorities.

There are now **two branches** after a model is trained (stage 2):

* the **priority / MINLP branch** (stages 3-7 below), which is where our own
  `MINLSearchExplainer` runs against declarative priority sets, and
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
├── minlp_test_config.yaml        <- stage 5: experiment configuration
├── minlp_runner.py               <- stage 6: MINLP search runner
├── random_runner.py              <- stage 7: random-search baseline runner
│
├── standard_methods/             <- baseline branch (runs after stage 2)
│   ├── predicted_dataset_setup.py       <- stage 2b: predicted-target dataset
│   ├── predicted_dataset_explorer.ipynb <- notebook: inspect predicted dataset
│   ├── selection.py                     <- sample/target + actionability logic
│   ├── methods.py                       <- regression CF method registry
│   ├── config.yaml                      <- standard-methods configuration
│   ├── runner.py                        <- runs the methods + writes results
│   ├── predicted_data/<dataset_key>/data.pkl
│   └── results/<dataset_key>/{samples.csv, counterfactuals.csv,
│                              metrics_summary.csv, summary.json}
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
configure → run methods**.

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
        params: {total_cfs: 5, method: genetic, backend: sklearn}
      - name: wachter
        params: {learning_rate: 0.1, max_iterations: 1000, proximity_weight: 0.005}
      - name: random_search
        params: {max_iterations: 3000, seed: 42}
      # ... other methods ...
```

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

For each `(sample, method)` pair the runner records the CF, checks validity
(`|prediction - target| <= epsilon`), and computes proximity / sparsity /
timing metrics.

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
* `counterfactuals.csv` — one row per `(sample_id, method)`: the generated CF
  (`cf__<feature>` columns) plus per-CF metrics: `cf_prediction`, `validity`,
  `abs_pred_error`, `l1`, `l2`, `n_changed`, `sparsity_fraction`,
  `iterations`, `time_seconds`, `error`.
* `metrics_summary.csv` — per-method averages: `validity_rate`,
  `avg_abs_pred_error`, `avg_l1`, `avg_l2`, `avg_n_changed`,
  `avg_sparsity_fraction`, `avg_iterations`, `avg_time_seconds` (distance /
  sparsity averages are taken over the *valid* CFs only).
* `summary.json` — the same per-method summary in machine-readable form.

Join `counterfactuals.csv` to `samples.csv` on `sample_id` to compare each CF
against its original instance.

> Metrics are computed in the scaled feature space and are intentionally a
> superset of the MINLP result metrics, so the two branches can be compared
> later. New metrics can be appended without breaking the existing columns.

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
