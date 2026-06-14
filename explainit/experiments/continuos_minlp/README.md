# Continuous-target MINLP experiment

This experiment runs `MINLSearchExplainer.find_counterfactuals` against
regression models (continuous target). It is split into small,
single-purpose stages so a dataset, model, or priority configuration can
be iterated on without touching the rest of the pipeline.

The `random_runner.py` script reproduces the same configuration with
`RandomSearchExplainer` so MINLP results can be compared against a
random-search baseline on identical priorities.

## Directory layout

```
explainit/experiments/continuos_minlp/
├── README.md                     <- this file
├── __init__.py
├── data_setup.py                 <- stage 1: dataset pickles
├── model_setup.py                <- stage 2: trained Keras models
├── priority_sets.py              <- stage 3: priority builders (edit me)
├── priorities_selection.py       <- stage 4: workbench for analyser plots
├── minlp_test_config.yaml        <- stage 5: experiment configuration
├── minlp_runner.py               <- stage 6: MINLP search runner
├── random_runner.py              <- stage 7: random-search baseline runner
│
├── data/<dataset_key>/data.pkl
├── models/<dataset_key>/model.keras
├── analysis/<dataset_key>/<sample_idx>_<target>/{coverage.txt, dataset/, priorities/, analysis_summary.txt}
└── results/<dataset_key>/<sample_idx>_<target>/{minlp.json, random.json}
```

## End-to-end flow

### 1. Prepare datasets

```bash
python -m explainit.experiments.continuos_minlp.data_setup
# or just one:
python -m explainit.experiments.continuos_minlp.data_setup --datasets diabetes
# overwrite existing pickles:
python -m explainit.experiments.continuos_minlp.data_setup --force
```

Each dataset gets:

* a standard-scaled `X_train` / `X_test`,
* a MinMax-scaled `y_train` / `y_test` (target lives in `[0, 1]`),
* the fitted `x_scaler` and `y_scaler` to invert predictions later,
* persisted to `data/<dataset_key>/data.pkl`.

Add a new dataset by registering a loader in `DATASETS` inside
`data_setup.py`.

### 2. Train models

```bash
python -m explainit.experiments.continuos_minlp.model_setup
python -m explainit.experiments.continuos_minlp.model_setup --datasets diabetes --epochs 80
python -m explainit.experiments.continuos_minlp.model_setup --force
```

Models land at `models/<dataset_key>/model.keras`. The default
architecture is a two-layer MLP; add a custom builder by registering it
in `MODEL_BUILDERS` inside `model_setup.py`.

### 3. Author priorities

Edit `priority_sets.py`. The file already ships with
`build_diabetes_default` as an example -- copy that pattern, register
the function under the appropriate `<dataset_key>` / `<set_name>` in
`PRIORITY_SETS`.

A priority builder has the signature:

```python
def my_builder(ctx: ExperimentContext, sample: np.ndarray, target_y: float) -> Dict[str, Any]:
    ...
    return {"numerical": {...}, "categorical": {...}}
```

Helpers `exponential_priority`, `linear_priority`, `constant_priority`,
and `numerical_entry` are provided to keep builders short.

### 4. Inspect priorities + dataset

Open `priorities_selection.py`, edit the `USER_*` constants:

```python
USER_DATASET = "diabetes"
USER_PRIORITY_SET = "default"
USER_SAMPLE_INDICES = (0, 5)
USER_TARGETS_SCALED = (0.25, 0.75)
```

Then run it:

```bash
python -m explainit.experiments.continuos_minlp.priorities_selection
# CLI overrides also work:
python -m explainit.experiments.continuos_minlp.priorities_selection \
    --dataset diabetes --priority-set default \
    --sample 0 --sample 5 --target 0.25 --target 0.75
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

Iterate by editing `priority_sets.py` and re-running this stage until
the coverage and visual feedback look right.

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
python -m explainit.experiments.continuos_minlp.minlp_runner
python -m explainit.experiments.continuos_minlp.minlp_runner --config minlp_test_config.yaml
python -m explainit.experiments.continuos_minlp.minlp_runner --dataset diabetes
```

Each pair produces `results/<dataset_key>/<sample_idx>_<target>/minlp.json`
with the headline metrics (validity, iterations, time) plus the resulting
counterfactual vector and the full per-iteration history.

### 7. Run random-search baseline

```bash
python -m explainit.experiments.continuos_minlp.random_runner
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
2. Optionally register a custom model builder in `model_setup.py`
   (`MODEL_BUILDERS`); otherwise the default MLP is used.
3. Add a builder in `priority_sets.py` and register it under
   `PRIORITY_SETS[<dataset_key>]["default"]`.
4. Add an entry to `experiments:` in `minlp_test_config.yaml`.
5. `data_setup` → `model_setup` → `priorities_selection` →
   `minlp_runner` → `random_runner`.
