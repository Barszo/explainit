# explainit

[![CI](https://github.com/Barszo/explainit_project/actions/workflows/ci.yml/badge.svg)](https://github.com/Barszo/explainit_project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Preference-aware counterfactual explanations for machine learning models.**

> *"What can I change to achieve my goal?"*

A **counterfactual explanation** tells a person how the input to a model would
have to change to get a different outcome: *"your loan would be approved if
your income were 4.1k instead of 3.0k"*. Most counterfactual methods return the
*closest* such change. But the closest change is not necessarily one the person
**can** make, or one they **prefer**: age cannot change, a raise may be
unrealistic, and someone may rather borrow less than wait a year.

explainit makes those preferences part of the search. You describe, per
feature, which values are allowed and how desirable each one is (a
**priority**), and explainit looks for counterfactuals that reach the model's
target while maximising that desirability.

> [!NOTE]
> explainit is research software under active development. The API may change
> between releases; see the [changelog](CHANGELOG.md).

## Installation

explainit requires Python 3.12 or newer. Install it from GitHub:

```bash
python -m pip install "git+https://github.com/Barszo/explainit_project.git@main"
```

NumPy, SciPy, and Matplotlib are installed automatically. For reproducible
work, pin a commit hash or release tag instead of `main`.

> [!IMPORTANT]
> This is **not** the unrelated `explainit` package on PyPI. Do not run
> `pip install explainit`.

## Quick start

A bank's model rejects a loan application. The applicant can earn a bit more
(but the less extra effort, the better), can borrow less (but would like to
keep as much of the loan as possible), and cannot change their age.

```python
import numpy as np

from explainit import RandomSearchExplainer, basic_linear


# Any function that maps a 2-D array (n_samples, n_features) to one prediction
# per row works as a model, e.g. ``lambda X: clf.predict_proba(X)[:, 1]``.
# Features: [monthly income (k EUR), loan amount (k EUR), age (years)]
def approval_probability(X):
    X = np.asarray(X, dtype=float)
    income, loan, age = X[:, 0], X[:, 1], X[:, 2]
    return 1 / (1 + np.exp(-(0.9 * income - 0.08 * loan + 0.02 * age - 2.0)))


applicant = np.array([3.0, 40.0, 35.0])  # approval probability: 0.14

priorities = {
    "numerical": {
        # Income may rise to 5k; the smaller the raise, the better (1 -> 0).
        0: {
            "min": 3.0,
            "max": 5.0,
            "function": lambda x: basic_linear(x, 3.0, 5.0, increasing=False),
        },
        # The loan may shrink to 10k; the larger the loan, the better (0 -> 1).
        1: {
            "min": 10.0,
            "max": 40.0,
            "function": lambda x: basic_linear(x, 10.0, 40.0, increasing=True),
        },
        # Age cannot change: no priority function means "keep this value".
        2: {"min": 35.0, "max": 35.0, "function": None},
    },
    "categorical": {},
}

explainer = RandomSearchExplainer(approval_probability, priorities, applicant, target=1)
counterfactuals, probabilities, scores, _ = explainer.generate_for_binary(
    expected_counterfactuals=3,
    target_class=1,
    threshold=0.5,
    n_candidates_per_cf=50,
    random_seed=0,
)

for cf, probability, score in zip(counterfactuals, probabilities, scores):
    print(f"income {cf[0]:.2f}k, loan {cf[1]:.1f}k -> p={probability:.2f}, preference {score:.2f}")
```

```text
income 3.20k, loan 19.1k -> p=0.51, preference 1.20
income 3.14k, loan 18.6k -> p=0.51, preference 1.22
income 3.11k, loan 18.1k -> p=0.51, preference 1.22
```

Instead of the minimal-distance answer, explainit suggests what fits this
applicant: a small income increase combined with a smaller loan, with age
untouched. The **preference score** is the sum of the per-feature priorities
(here at most 2.0), so higher is better.

`get_preference_breakdown` shows where the score comes from:

```python
breakdown = explainer.get_preference_breakdown(counterfactuals[0])
print({i: round(f["weight"], 2) for i, f in breakdown["numerical"].items() if f["actionable"]})
```

```text
{0: 0.9, 1: 0.3}
```

## Core concepts

| Term | Meaning |
| --- | --- |
| **model** (`model_pred`) | Callable that takes a 2-D array of samples and returns one prediction per row (a value or a class probability). |
| **sample** | The original input you want to explain (1-D array). |
| **target** | The desired prediction. For regression-style search a counterfactual is valid when `abs(model(x) - target) <= epsilon`; for binary search when the probability crosses `threshold`. |
| **priority** | For each feature: which values are allowed, and how desirable each allowed value is, on a 0-to-1 scale. |
| **preference score** | Sum of the priorities of a counterfactual's feature values. Used to rank valid counterfactuals. |

Features are referred to by **column index**, in the same (preprocessed) space
the model receives. Results are therefore in model units: if your model expects
standardised features, so do the priorities.

### Numerical priorities

```python
from explainit import exponential

numerical = {
    # Allowed range, plus how desirable each value in it is (must return 0..1).
    0: {"min": 0.0, "max": 1.0, "function": lambda x: exponential(x, 0.4, 0.9)},
    # Immutable feature: no function, fixed at this value.
    1: {"min": 0.5, "max": 0.5, "function": None},
}
```

Built-in priority functions ramp between `x0` and `x1`, from 0 to 1
(`increasing=True`) or from 1 to 0 (`increasing=False`):

| Function | Shape |
| --- | --- |
| `basic_linear(x, x0, x1, increasing=True)` | Linear ramp. |
| `exponential(x, x0, x1, increasing=True, a=5)` | Convex ramp; larger `a` makes it steeper. |
| `basic_linear_step(x, x0, x1, increasing=True)` | Linear ramp (same as `basic_linear`, kept for compatibility). |

Any **vectorised** function returning values in `[0, 1]` works, for example a
preference peaked around 4.0:

```python
numerical[2] = {"min": 2.0, "max": 6.0, "function": lambda x: np.exp(-(((x - 4.0) / 0.5) ** 2))}
```

### Categorical priorities

One-hot encoded features are grouped by the tuple of their column indices. Each
allowed combination gets a weight; a weight of `0` or `None` forbids it.

```python
categorical = {
    # Columns 3 and 4 encode housing: (1, 0) = owner, (0, 1) = renter.
    (3, 4): {(1, 0): 1.0, (0, 1): 0.4},
}
# Combined: {"numerical": numerical, "categorical": categorical}
```

## Choosing an explainer

| | `RandomSearchExplainer` | `MINLPSearchExplainer` |
| --- | --- | --- |
| Idea | Samples candidates from the priority-weighted space and keeps valid ones with the best score. | Builds a local Shapley-value surrogate around a reference point and optimises the priorities with SLSQP, re-checking with the real model. |
| Needs | Model, sample, target, priorities. | Same, plus a reference **dataset** (e.g. training data). |
| Returns | Several diverse counterfactuals. | One optimised counterfactual, plus diagnostics in `last_search_result`. |
| Good for | Quick exploration, any model, many alternatives. | Higher-quality single answers with fewer model calls. |

Both explainers support regression-style targets and binary classification
(`generate_for_binary` / `find_counterfactuals_for_binary`), and accept a
`random_seed` for reproducible results.

### MINLP example

Continuing the loan example, ask for an approval probability of 0.6:

```python
from explainit import MINLPSearchExplainer

rng = np.random.default_rng(0)
past_applicants = np.column_stack(
    [rng.uniform(2, 6, 300), rng.uniform(5, 60, 300), rng.uniform(20, 65, 300)]
)

minlp = MINLPSearchExplainer(
    approval_probability,
    priorities,  # the priorities from the quick start
    applicant,
    target=0.6,
    dataset=past_applicants,
    epsilon=0.02,
)
counterfactual = minlp.find_counterfactuals(random_seed=0)

print(np.round(counterfactual, 2), approval_probability([counterfactual]).round(2))
print(minlp.last_search_result["reached_target"])
```

```text
[ 3.1  13.49 35.  ] [0.6]
True
```

## Visualising priorities

```python
from explainit import plot_priorities

plot_priorities(
    priorities,
    sample=applicant,
    feature_names=["income", "loan", "age"],
    show=True,  # or save_dir="figures" to write PNG files
)
```

`RandomSearchExplainer.investigate_probability_distribution()` additionally
shows how the random search samples each feature.

## Logging

explainit logs its search progress to the `explainit` logger but never
configures logging itself. To see the messages:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Limitations

- Priorities are specified by hand and combined additively (a sum over
  features); interactions between preferences are not modelled.
- Counterfactuals are expressed in the model's input space. Map them back to
  original units yourself if the model uses preprocessed features.
- A counterfactual shows how the *model* responds, not what causes the outcome
  in the real world.
- Search is heuristic: failing to find a counterfactual does not prove that
  none exists.

## Learn more

- [How explainit works](docs/how-it-works.md): the two search methods, their
  stages, and how to read the search diagnostics.

## Project layout

```text
explainit/
├── explainers/   # RandomSearchExplainer, MINLPSearchExplainer
├── priorities/   # built-in priority functions
└── utils/        # plotting helpers
docs/             # additional documentation
tests/            # pytest suite, including the examples in the docs
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, coding standards and pull request process, and please follow
our [Code of Conduct](CODE_OF_CONDUCT.md).

## Citing

If you use explainit in research, please cite it; see [CITATION.cff](CITATION.cff)
(GitHub shows a "Cite this repository" button). The method was presented in:

> B. Szostak and R. Doroz, *Designing Preference-Aware Counterfactual
> Explanations for Adaptive Information Systems*, Information Systems
> Development (ISD 2026), Prague, 2026.

## License

[MIT](LICENSE) © Bartosz Szostak
