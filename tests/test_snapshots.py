"""Snapshot ("characterisation") tests.

Each scenario runs an explainer and compares the outputs, plus the number of
model calls and rows, with a recording in ``tests/snapshots/results.json``.
Refactoring that should not change behaviour must keep these identical.

Optimiser results depend on the NumPy/SciPy versions, so the comparison only
runs with the versions the snapshots were recorded with (CI has a dedicated
job for this); otherwise the tests are skipped.

After an intentional behaviour change, re-record with::

    pytest tests/test_snapshots.py --update-snapshots
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
from problems import CountingModel

from explainit import MINLPSearchExplainer, RandomSearchExplainer, basic_linear, exponential

SNAPSHOT_FILE = Path(__file__).parent / "snapshots" / "results.json"

pytestmark = pytest.mark.snapshot


def _versions():
    return {
        "numpy": ".".join(np.__version__.split(".")[:2]),
        "scipy": ".".join(scipy.__version__.split(".")[:2]),
    }


def _ramp(low, high, increasing=True, kind="linear"):
    fn = basic_linear if kind == "linear" else exponential
    return lambda x: fn(x, low, high, increasing=increasing)


def _numerical_problem():
    def model(X):
        return X[:, 0] * 0.8 + X[:, 1] * 0.2

    priorities = {
        "numerical": {
            0: {"min": 0.0, "max": 1.0, "function": _ramp(0.4, 0.9, kind="exp")},
            1: {"min": 0.0, "max": 1.0, "function": _ramp(0.2, 0.8, kind="exp")},
        },
        "categorical": {},
    }
    return model, priorities, np.array([0.2, 0.4])


def _mixed_model(X):
    return X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3] + 0.1 * np.sin(4 * X[:, 4])


def _mixed_priorities(fixed_feature=False):
    priorities = {
        "numerical": {
            0: {"min": 0.0, "max": 1.0, "function": _ramp(0.2, 0.8)},
            1: {"min": 0.0, "max": 1.0, "function": _ramp(0.1, 0.7, increasing=False)},
            4: {"min": 0.0, "max": 1.0, "function": _ramp(0.0, 1.0, kind="exp")},
        },
        "categorical": {(2, 3): {(1.0, 0.0): 1.0, (0.0, 1.0): 0.5}},
    }
    if fixed_feature:
        priorities["numerical"][4] = {"min": 0.3, "max": 0.3, "function": None}
    return priorities


MIXED_SAMPLE = np.array([0.1, 0.6, 0.0, 1.0, 0.3])


def _mixed_dataset(n_rows):
    rng = np.random.default_rng(0)
    dataset = rng.uniform(0, 1, size=(n_rows, 5))
    category = rng.integers(0, 2, size=n_rows)
    dataset[:, 2], dataset[:, 3] = category, 1 - category
    return dataset


def _sigmoid(fn, shift):
    return lambda X: 1.0 / (1.0 + np.exp(-6.0 * (fn(X) - shift)))


def _random_search(model, priorities, sample, target, method, **kwargs):
    counted = CountingModel(model)
    explainer = RandomSearchExplainer(counted, priorities, sample, target)
    cfs, preds, scores, iterations = getattr(explainer, method)(**kwargs)
    return {
        "counterfactuals": np.asarray(cfs, dtype=float).tolist(),
        "predictions": np.asarray(preds, dtype=float).tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
        "iterations": [int(i) for i in iterations],
        "model_calls": counted.calls,
        "model_rows": counted.rows,
    }


def _minlp(model, priorities, sample, target, dataset, **kwargs):
    counted = CountingModel(model)
    explainer = MINLPSearchExplainer(counted, priorities, sample, target, dataset)
    cf = explainer.find_counterfactuals(**kwargs)
    result = explainer.last_search_result
    return {
        "counterfactual": None if cf is None else np.asarray(cf, dtype=float).tolist(),
        "reached_target": bool(result["reached_target"]),
        "stop_reason": result["stop_reason"],
        "iterations_run": int(result["iterations_run"]),
        "exemplar_source": explainer.exemplar_source,
        "cf_source": result["cf_source"],
        "priority_score": float(result["priority_score"]),
        "model_calls": counted.calls,
        "model_rows": counted.rows,
    }


def _minlp_binary(model, priorities, sample, dataset, **kwargs):
    counted = CountingModel(model)
    explainer = MINLPSearchExplainer(counted, priorities, sample, 1, dataset)
    cfs, preds, scores, counts = explainer.find_counterfactuals_for_binary(**kwargs)
    return {
        "counterfactuals": np.asarray(cfs, dtype=float).tolist(),
        "predictions": np.asarray(preds, dtype=float).tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
        "counts": [int(c) for c in counts],
        "model_calls": counted.calls,
        "model_rows": counted.rows,
    }


def scenarios():
    """Every scenario builds fresh inputs, so no run can affect another."""
    num = _numerical_problem
    mixed = _mixed_priorities
    return {
        "random/regression-monte-carlo": lambda: _random_search(
            num()[0], num()[1], num()[2], 0.9, "generate_random_samples",
            expected_counterfactuals=3, max_iterations=5000, epsilon=0.05, random_seed=42,
        ),
        "random/regression-uniform": lambda: _random_search(
            num()[0], num()[1], num()[2], 0.9, "generate_random_samples",
            expected_counterfactuals=3, max_iterations=5000, epsilon=0.05, random_seed=5,
            use_monte_carlo=False,
        ),
        "random/regression-mixed-best-of-20": lambda: _random_search(
            _mixed_model, mixed(), MIXED_SAMPLE, 1.2, "generate_random_samples",
            expected_counterfactuals=4, max_iterations=5000, epsilon=0.05, random_seed=3,
            n_candidates_per_cf=20, return_top_n=2,
        ),
        "random/regression-fixed-feature": lambda: _random_search(
            _mixed_model, mixed(fixed_feature=True), MIXED_SAMPLE, 1.2,
            "generate_random_samples",
            expected_counterfactuals=2, max_iterations=5000, epsilon=0.05, random_seed=8,
        ),
        "random/binary": lambda: _random_search(
            _sigmoid(_mixed_model, 1.2), mixed(), MIXED_SAMPLE, 1, "generate_for_binary",
            expected_counterfactuals=3, max_iterations=3000, random_seed=0,
            n_candidates_per_cf=10,
        ),
        "minlp/default": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, 1.2, _mixed_dataset(400), random_seed=1
        ),
        "minlp/approximate-shapley": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, 1.2, _mixed_dataset(400),
            random_seed=2, shap_approx=True, num_samples=50,
        ),
        "minlp/no-trust-region": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, 1.2, _mixed_dataset(400),
            random_seed=1, trust_region=False,
        ),
        "minlp/no-corrections": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, 1.2, _mixed_dataset(400),
            random_seed=1, residual_correction=False, restoration=False, peak_lock_in=False,
        ),
        "minlp/fixed-feature": lambda: _minlp(
            _mixed_model, mixed(fixed_feature=True), MIXED_SAMPLE, 1.0, _mixed_dataset(400),
            random_seed=1,
        ),
        "minlp/random-anchor": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, 1.2, _mixed_dataset(30), random_seed=4
        ),
        "minlp/numerical-only": lambda: _minlp(
            num()[0], num()[1], num()[2], 0.9,
            np.random.default_rng(1).uniform(0, 1, (400, 2)), random_seed=0,
        ),
        "minlp/binary": lambda: _minlp_binary(
            _sigmoid(_mixed_model, 1.2), mixed(), MIXED_SAMPLE, _mixed_dataset(400),
            target_class=1, threshold=0.5,
        ),
    }  # fmt: skip


SCENARIOS = scenarios()


def _run(name):
    logging.disable(logging.CRITICAL)
    try:
        return SCENARIOS[name]()
    finally:
        logging.disable(logging.NOTSET)


def _load():
    if SNAPSHOT_FILE.exists():
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    return {"versions": None, "results": {}}


def _assert_matches(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: length differs"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_matches(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-7, abs=1e-9), path
    else:
        assert actual == expected, path


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_snapshot(name, request):
    if request.config.getoption("--update-snapshots"):
        data = _load()
        data["versions"] = {**_versions(), "python": sys.version.split()[0]}
        data["results"][name] = _run(name)
        SNAPSHOT_FILE.parent.mkdir(exist_ok=True)
        ordered = dict(sorted(data["results"].items()))
        text = json.dumps({**data, "results": ordered}, indent=1, sort_keys=False)
        SNAPSHOT_FILE.write_text(text + "\n", encoding="utf-8")
        return

    data = _load()
    recorded = data["versions"] or {}
    current = _versions()
    if any(recorded.get(lib) != current[lib] for lib in current):
        pytest.skip(f"snapshots recorded with {recorded}, running {current}")
    assert name in data["results"], "no snapshot recorded; run with --update-snapshots"
    _assert_matches(_run(name), data["results"][name])
