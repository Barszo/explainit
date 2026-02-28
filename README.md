# explainit

**explainit** is a Python library for generating *counterfactual explanations* for machine learning models. Given a model's prediction for a specific input, explainit finds the minimal changes to that input that would flip the prediction to a desired target — answering the question *"What would need to be different for the model to predict X instead?"*

---

## What is a Counterfactual Explanation?

A counterfactual explanation describes an alternative input that leads to a different (desired) model output, while changing as few features as possible. For example:

> *"Your loan application was denied. If your annual income were €5,000 higher and your credit score were above 680, it would have been approved."*

These explanations are actionable, human-readable, and useful for debugging models and communicating decisions to end users.

---

## Key Features

- **Multiple search strategies** for finding counterfactuals:
  - **Basic (gradient-based)** – iteratively steps numerical features toward the target prediction
  - **Random Search** – Monte Carlo sampling guided by user-defined priority functions
  - **MINLP Search** – Mixed-Integer Nonlinear Programming optimization that uses Shapley values to rank the most impactful features before searching

- **Actionability constraints** – mark individual features as non-modifiable (e.g. age, gender) to ensure only realistic changes are suggested

- **Priority functions** for numerical and categorical features – control *how* features should change using linear or nonlinear (exponential) weighting functions that encode domain knowledge

- **Shapley value attribution** – compute exact or approximate Shapley values to identify which features contribute most to the difference between the original and target prediction

- **Visualization tools** – plot priority functions and empirical/theoretical sampling distributions to inspect and debug the explanation search

---

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.13, [SHAP](https://github.com/shap/shap), NumPy, SciPy, Matplotlib, scikit-learn

---

## Quick Start

### Random Search Explainer

```python
from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.linear import basic_linear

# Define priority functions for each feature
priorities = {
    'numerical': {
        0: {'function': lambda x: basic_linear(x, 30000, 80000), 'min': 30000, 'max': 80000},
        1: 0,  # feature 1 is not actionable (fixed at sample value)
    },
    'categorical': {
        (2, 3): {(0, 1): 0.8, (1, 0): 0.2, (0, 0): 0},  # (0,0) is forbidden
    }
}

explainer = RandomSearchExplainer(
    model_pred=model.predict,
    priorities=priorities,
    sample=my_sample,
    target=1.0,
)

# Generate 500 counterfactual candidates within 5% of the target prediction
samples, predictions = explainer.generate_random_samples(n_samples=500, epsilon=0.05)

# Visualize the priority and sampling distributions
explainer.display_priorities()
explainer.investigate_probability_distribution()
```

### MINLP Search Explainer

```python
from explainit.explainers.minlp_search import MINLSearchExplainer

explainer = MINLSearchExplainer(
    model_pred=model.predict,
    priorities=priorities,
    sample=my_sample,
    target=1.0,
    dataset=X_train,
)

# Compute Shapley values to identify the most influential features
shapley_vals = explainer.calc_shapley(my_sample)
```

---

## Priority Functions

Priority functions encode the desirability of each feature value and are used to guide the search. They map a feature value to a weight in `[0, 1]`.

```python
from explainit.priorities.linear import basic_linear
from explainit.priorities.nonlinear import exponential

# Linear ramp: 0 below x0=30k, 1 above x1=80k (increasing)
f_linear = lambda x: basic_linear(x, x0=30000, x1=80000, increasing=True)

# Exponential ramp with steepness a=5
f_exp = lambda x: exponential(x, x0=30000, x1=80000, increasing=True, a=5)
```

A priority of `0` for a numerical feature marks it as **unactionable** — the feature is fixed at its original value and will not be changed.

---

## Project Structure

```
explainit/
├── explainers/
│   ├── basic.py           # Gradient-based explainer
│   ├── random_search.py   # Monte Carlo / random search explainer
│   └── minlp_search.py    # MINLP optimization explainer
├── priorities/
│   ├── linear.py          # Linear priority functions
│   └── nonlinear.py       # Nonlinear (exponential) priority functions
├── utils/
│   ├── plot_styles.py     # Visualization helpers
│   └── priorities_utils.py
└── examples/              # Example notebooks and scripts
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.