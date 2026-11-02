# How explainit works

This page summarises the two search methods at the level needed to use them
well and to find your way around the code. Notation follows the README: a
model $f$, an original sample $\mathbf{x}$, a target $y^*$, a tolerance
$\varepsilon$, and per-feature priorities $p_i(x_i) \in [0, 1]$.

A candidate $\mathbf{x}'$ is a **valid counterfactual** when
$|f(\mathbf{x}') - y^*| \le \varepsilon$ (or, for binary search, when the
predicted probability is on the requested side of the threshold). Among valid
candidates, explainit prefers the one with the highest **preference score**

$$S(\mathbf{x}') = \sum_{i \in \text{numerical}} p_i(x'_i) + \sum_{g \in \text{categorical}} w_g(\mathbf{x}'_g).$$

Priorities therefore play two roles: they **restrict** the search (values
outside `[min, max]`, forbidden categories and immutable features are never
proposed; the MINLP search also excludes zero-priority regions) and they
**rank** the remaining options.

## Random search (`RandomSearchExplainer`)

Source: [`explainit/explainers/random_search.py`](../explainit/explainers/random_search.py)

1. Immutable features are fixed at their sample value; forbidden categories
   are removed.
2. Each candidate is sampled feature by feature. With `use_monte_carlo=True`
   (default) numerical values are drawn with probability proportional to their
   priority (rejection sampling), and categorical combinations proportional to
   their weight; otherwise sampling is uniform.
3. The model checks each candidate. For every requested counterfactual the
   search collects `n_candidates_per_cf` valid candidates and keeps the one with
   the highest preference score.

The method is model-agnostic and simple, and it returns diverse alternatives.
Its cost is the number of model calls, bounded by `max_iterations` per
counterfactual.

## Shapley-guided MINLP search (`MINLPSearchExplainer`)

Source: [`explainit/explainers/minlp/`](../explainit/explainers/minlp/)

The search treats counterfactual generation as a mixed-integer nonlinear
program: maximise $S(\mathbf{x}')$ subject to reaching the target, with
categorical features as discrete choices. The true model is replaced locally by
a linear surrogate built from Shapley values, and every candidate is re-checked
with the real model. The stages match the `[Stage N]` messages in the log:

| Stage | What happens | Module |
| --- | --- | --- |
| 0 | Derive each numerical feature's allowed region from where its priority is positive. | `bounds.py` |
| 1 | Pick an **anchor**: a dataset row that respects the priorities and whose prediction is close to the target (falls back to random sampling of the allowed region). | `anchor.py` |
| 3 | Compute Shapley values between the current point and the anchor (exact, or Monte-Carlo with `shap_approx=True`), giving a linear surrogate $h(\mathbf{x}')$. | `shapley.py`, `surrogate.py` |
| 4 | Prune categorical combinations that cannot help, then solve one linear program per combination for a feasible warm start. | `combinations.py`, `linear_program.py`, `warm_start.py` |
| 5 | Optimise the preference score with SLSQP subject to $\lvert h(\mathbf{x}') - y^*\rvert \le \varepsilon$, inside a trust region. | `optimizer.py`, `bounds.py` (`TrustRegion`) |
| 6 | Evaluate the candidate with the real model, apply residual correction or restoration if it misses the band, and re-linearise around the new point. | `refinement.py`, `incumbent.py` |
| 7 | **Peak lock-in**: push features towards their highest-priority values while other features compensate, keeping the prediction in the band. | `refinement.py` |

`explainer.py` holds `MINLPSearchExplainer`, which runs the stages in this
order; `scoring.py` computes preference scores, `intervals.py` handles allowed
intervals, and `binary.py` contains the separate search for binary
classifiers.

Stages 3 to 6 repeat for up to `max_iterations` passes, stopping early after
`patience` passes without improvement. The best valid candidate is returned;
if none was found, the closest candidate is returned (or `None` with
`return_when_fails=False`). If no anchor can be found at all (for example,
the target is outside what the allowed region can reach), an exception is
raised.

After a run, `last_search_result` explains what happened:

```python
import numpy as np

from explainit import MINLPSearchExplainer, basic_linear


def model(X):
    X = np.asarray(X, dtype=float)
    return X[:, 0] + 0.5 * X[:, 1]


priorities = {
    "numerical": {
        0: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.2, 0.8)},
        1: {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.1, 0.7)},
    },
    "categorical": {},
}
dataset = np.random.default_rng(0).uniform(0, 1, size=(50, 2))

explainer = MINLPSearchExplainer(model, priorities, np.array([0.1, 0.2]), 1.0, dataset)
explainer.find_counterfactuals(random_seed=0)

result = explainer.last_search_result
print(result["reached_target"], result["distance"] <= explainer.epsilon)
```

```text
True True
```

Useful keys include `stop_reason`, `distance` (to the target), `iterations_run`,
`cf_source` (`anchor` or `optimiser`), `priority_score`, `history` (one entry
per pass) and `step_control` (trust-region statistics). The anchor's origin
is in `explainer.exemplar_source`.

## Reproducibility

Both explainers accept `random_seed`. A seeded run uses private random number
generators, so repeated runs give identical results and the global NumPy and
`random` state is not modified.

Random search results are identical on every machine. MINLP results are
identical on the same operating system with the same NumPy/SciPy versions;
across operating systems the optimiser can take a slightly different path, so
the counterfactual may differ in the third or fourth decimal place.
