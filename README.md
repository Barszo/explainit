# explainit

Lightweight workspace for experimenting with preference-based counterfactual explanations.

After the recent cleanup, the active repository is much smaller than before: the maintained code now lives in the `explainit/` package, while older notebooks, experiments, datasets, and notes have been moved out of the main project flow into `trash/`.

## Current repository status

- **Active code**: core explainers, priority functions, plotting, and analysis utilities
- **Archived material**: older notebooks, experiments, model artifacts, and reference notes in `trash/`
- **Tests**: `tests/` is currently only a placeholder package

## Active project structure

```text
explainit_project/
├── explainit/
│   ├── __init__.py
│   ├── logging_config.py
│   ├── explainers/
│   │   ├── minlp_search.py
│   │   └── random_search.py
│   ├── priorities/
│   │   ├── linear.py
│   │   └── nonlinear.py
│   └── utils/
│       ├── dataset_analyzer.py
│       ├── plot_styles.py
│       ├── priorities_analyser.py
│       └── priority_plots.py
├── tests/
├── trash/
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
└── setup.py
```

## What is currently included

### Explainers

- **`RandomSearchExplainer`** (`explainit/explainers/random_search.py`)
  - samples candidate points from user-defined numerical and categorical priorities
  - supports regression-style targets and binary classification thresholds
  - includes preference scoring, per-feature breakdowns, and probability-distribution visualizations

- **`MINLSearchExplainer`** (`explainit/explainers/minlp_search.py`)
  - iterative counterfactual search based on Shapley re-linearisation
  - includes bound derivation from priority functions, trust-region control, restoration, and priority-guided candidate selection
  - exposes priority plotting for the configured search space

### Priority functions

- **`explainit/priorities/linear.py`**
  - `basic_linear`

- **`explainit/priorities/nonlinear.py`**
  - `exponential`
  - `basic_linear_step`

These functions return values in the `[0, 1]` range and are used to encode how desirable different feature values are.

### Utilities

- **`priority_plots.py`**
  - plots numerical and categorical priorities
  - plots sampling-oriented probability distributions
  - can save generated figures to disk

- **`dataset_analyzer.py`**
  - analyzes datasets feature-by-feature
  - infers feature/target types
  - produces summary text and plots for distributions and correlations

- **`priorities_analyser.py`**
  - reports how configured priorities cover the dataset
  - finds closest exemplars for target predictions
  - combines coverage reporting with dataset and priority visualizations

## Installation

For the current cleaned-down repo, the most reliable setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`CONTRIBUTING.md` currently uses the same editable install command:

```bash
pip install -e .
```

## Minimal usage example

```python
import numpy as np

from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.nonlinear import exponential


def model_pred(X):
    X = np.asarray(X)
    return X[:, 0] * 0.8 + X[:, 1] * 0.2


sample = np.array([0.2, 0.4])
target = 0.9

priorities = {
    "numerical": {
        0: {"min": 0.0, "max": 1.0, "function": lambda x: exponential(x, 0.4, 0.9, increasing=True)},
        1: {"min": 0.0, "max": 1.0, "function": lambda x: exponential(x, 0.2, 0.8, increasing=True)},
    },
    "categorical": {},
}

explainer = RandomSearchExplainer(
    model_pred=model_pred,
    priorities=priorities,
    sample=sample,
    target=target,
)

counterfactuals, predictions, scores, iterations = explainer.generate_random_samples(
    expected_counterfactuals=3,
    max_iterations=5000,
    epsilon=0.05,
    random_seed=42,
)
```

## Archived material

The `trash/` directory now acts as a holding area for material removed from the active repo structure, including:

- notebooks and exploratory scripts
- older experiment folders
- model-training assets and datasets
- draft documentation and theory notes

That content may still be useful as reference, but it should be treated as archived rather than part of the maintained package surface.

## Notes

- The README no longer documents the old examples/experiments/model folders as active top-level project components, because they are no longer part of the cleaned main structure.
- If you are rebuilding the project from this slimmer base, start from `explainit/` and ignore `trash/` unless you need historical context.
