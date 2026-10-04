# explainit

`explainit` is a Python package for building preference-based counterfactual explanations for machine learning models. It lets you describe which feature changes are more desirable, sample or optimize candidate counterfactuals, and visualize both the preference structure and the data around it.

## Project structure

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
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
└── setup.py
```

## Main components

### Explainers

#### `RandomSearchExplainer`

Defined in `explainit/explainers/random_search.py`.

Use it when you want to generate counterfactual candidates by sampling from user-defined priorities.

Current capabilities:
- handles numerical and categorical priorities
- supports regression-style targets with an `epsilon` tolerance
- supports binary-classification search with a decision threshold
- computes aggregate preference scores for found candidates
- provides per-feature preference breakdowns
- can visualize priorities and sampling distributions

#### `MINLSearchExplainer`

Defined in `explainit/explainers/minlp_search.py`.

Use it when you want a more optimization-driven search procedure based on iterative Shapley re-linearisation.

Current capabilities:
- derives feasible bounds from priority functions
- uses exemplar-based search initialization
- supports trust-region updates and residual correction
- restores near-feasible candidates back into the target band
- scores and selects candidates using configured priorities
- can visualize configured priorities

### Priority functions

Priority functions describe how desirable feature values are. They are used in the `priorities["numerical"]` configuration and are expected to return values in the `[0, 1]` range.

Available helpers:

- `explainit.priorities.linear.basic_linear`
- `explainit.priorities.nonlinear.exponential`
- `explainit.priorities.nonlinear.basic_linear_step`

### Utilities

#### `priority_plots.py`

Plotting helpers for:
- numerical priority functions
- categorical priority mappings
- probability-style views of sampling behavior

#### `dataset_analyzer.py`

Dataset diagnostics for:
- feature and target type inference
- per-feature descriptive summaries
- feature distributions
- correlation plots and target relationships

#### `priorities_analyser.py`

Analysis helpers for:
- checking how priorities cover a dataset
- locating dataset rows closest to target predictions
- generating combined priority and dataset diagnostics

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## How priorities are structured

Both explainers expect a priorities dictionary with `numerical` and `categorical` sections.

### Numerical priorities

Each actionable numerical feature is typically defined as:

```python
{
    feature_index: {
        "min": lower_bound,
        "max": upper_bound,
        "function": priority_function,
    }
}
```

Example:

```python
from explainit.priorities.nonlinear import exponential

priorities = {
    "numerical": {
        0: {
            "min": 0.0,
            "max": 1.0,
            "function": lambda x: exponential(x, 0.4, 0.9, increasing=True),
        }
    },
    "categorical": {},
}
```

### Categorical priorities

Categorical features are grouped by tuples of column indices, and each allowed category combination is assigned a weight.

Example:

```python
priorities = {
    "numerical": {},
    "categorical": {
        (2, 3): {
            (1, 0): 1.0,
            (0, 1): 0.4,
        }
    },
}
```

## Usage

### 1. Random search explainer

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

Useful methods:
- `generate_random_samples(...)`
- `generate_for_binary(...)`
- `calculate_preference_score(sample)`
- `get_preference_breakdown(sample)`
- `display_priorities(...)`
- `investigate_probability_distribution(...)`

### 2. MINLP-style explainer

```python
import numpy as np

from explainit.explainers.minlp_search import MINLSearchExplainer
from explainit.priorities.linear import basic_linear


def model_pred(X):
    X = np.asarray(X)
    return X[:, 0] + 0.5 * X[:, 1]


dataset = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.6, 0.5],
    [0.8, 0.9],
])

sample = np.array([0.1, 0.2])
target = 1.0

priorities = {
    "numerical": {
        0: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.2, 0.8, increasing=True)},
        1: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.1, 0.7, increasing=True)},
    },
    "categorical": {},
}

explainer = MINLSearchExplainer(
    model_pred=model_pred,
    priorities=priorities,
    sample=sample,
    target=target,
    dataset=dataset,
)

counterfactual = explainer.find_counterfactuals()
```

Useful methods:
- `find_counterfactuals(...)`
- `find_counterfactuals_for_binary(...)`
- `display_priorities(...)`

### 3. Plot priorities directly

```python
from explainit.utils.priority_plots import plot_priorities

plot_priorities(priorities, sample=sample, show=True)
```

### 4. Analyze a dataset

```python
from pathlib import Path

from explainit.utils.dataset_analyzer import analyze_dataset

report = analyze_dataset(
    X=dataset,
    y=model_pred(dataset),
    feature_names=["feature_0", "feature_1"],
    dataset_key="demo_dataset",
    output_dir=Path("analysis_output"),
)
```

### 5. Analyze priorities against a dataset

```python
from pathlib import Path

from explainit.utils.priorities_analyser import analyse_priorities

report = analyse_priorities(
    model=model_pred,
    dataset=dataset,
    target_values=[target],
    priorities=priorities,
    feature_names=["feature_0", "feature_1"],
    output_dir=Path("priority_analysis_output"),
)
```

## Development

Editable installation:

```bash
pip install -e .
```

The package version is defined in `explainit/__init__.py`.
