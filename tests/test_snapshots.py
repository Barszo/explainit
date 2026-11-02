"""Snapshot ("characterisation") tests.

Each scenario runs an explainer and compares the outputs, plus the number of
model calls and rows, with a recording in ``tests/snapshots/<platform>.json``.
Refactoring that should not change behaviour must keep these identical.

Optimiser results depend on floating-point details of the platform and on the
NumPy/SciPy versions, so there is one recording per operating system, and the
comparison only runs with the versions it was recorded with (CI has a
dedicated Linux job); otherwise the tests are skipped.

After an intentional behaviour change, re-record with::

    pytest tests/test_snapshots.py --update-snapshots

The CI snapshot job also records a fresh ``linux.json`` and uploads it as the
``snapshots-linux`` artifact, so it can be updated without a Linux machine.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
from problems import CountingModel

from explainit import (
    MINLPSearchExplainer,
    MINLPSettings,
    RandomSearchExplainer,
    basic_linear,
    exponential,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

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


def _random_search(model, priorities, sample, settings=None, **explain_kwargs):
    counted = CountingModel(model)
    explainer = RandomSearchExplainer(counted, priorities, **(settings or {}))
    result = explainer.explain(sample, **explain_kwargs)
    assert result.valid.all()
    return {
        "counterfactuals": result.counterfactuals.tolist(),
        "predictions": result.predictions.tolist(),
        "scores": result.scores.tolist(),
        "iterations": [int(i) for i in result.diagnostics["iterations"]],
        "model_calls": counted.calls,
        "model_rows": counted.rows,
    }


def _minlp(model, priorities, sample, dataset, settings=None, **explain_kwargs):
    counted = CountingModel(model)
    explainer = MINLPSearchExplainer(
        counted, priorities, dataset, settings=MINLPSettings(**(settings or {}))
    )
    result = explainer.explain(sample, **explain_kwargs)
    d = result.diagnostics
    assert list(result.valid) == [d["reached_target"]] * len(result)
    return {
        "counterfactual": result.counterfactuals[0].tolist() if len(result) else None,
        "reached_target": bool(d["reached_target"]),
        "stop_reason": d["stop_reason"],
        "iterations_run": int(d["iterations_run"]),
        "exemplar_source": d["exemplar_source"],
        "cf_source": d["cf_source"],
        "priority_score": float(d["priority_score"]),
        "model_calls": counted.calls,
        "model_rows": counted.rows,
    }


def _minlp_binary(model, priorities, sample, dataset, **explain_kwargs):
    counted = CountingModel(model)
    result = MINLPSearchExplainer(counted, priorities, dataset).explain(sample, **explain_kwargs)
    return {
        "counterfactuals": result.counterfactuals.tolist(),
        "predictions": result.predictions.tolist(),
        "scores": result.scores.tolist(),
        "counts": [int(c) for c in result.diagnostics["combination_numbers"]],
        "model_calls": counted.calls,
        "model_rows": counted.rows,
    }


def scenarios():
    """Every scenario builds fresh inputs, so no run can affect another."""
    num = _numerical_problem
    mixed = _mixed_priorities
    return {
        "random/regression-monte-carlo": lambda: _random_search(
            num()[0], num()[1], num()[2], settings={"max_iterations": 5000},
            target=0.9, tolerance=0.05, n_counterfactuals=3, random_seed=42,
        ),
        "random/regression-uniform": lambda: _random_search(
            num()[0], num()[1], num()[2],
            settings={"max_iterations": 5000, "use_monte_carlo": False},
            target=0.9, tolerance=0.05, n_counterfactuals=3, random_seed=5,
        ),
        "random/regression-mixed-best-of-20": lambda: _random_search(
            _mixed_model, mixed(), MIXED_SAMPLE,
            settings={"max_iterations": 5000, "candidates_per_counterfactual": 20},
            target=1.2, tolerance=0.05, n_counterfactuals=4, top_n=2, random_seed=3,
        ),
        "random/regression-fixed-feature": lambda: _random_search(
            _mixed_model, mixed(fixed_feature=True), MIXED_SAMPLE,
            settings={"max_iterations": 5000},
            target=1.2, tolerance=0.05, n_counterfactuals=2, random_seed=8,
        ),
        "random/binary": lambda: _random_search(
            _sigmoid(_mixed_model, 1.2), mixed(), MIXED_SAMPLE,
            settings={"max_iterations": 3000, "candidates_per_counterfactual": 10},
            target_class=1, n_counterfactuals=3, random_seed=0,
        ),
        "minlp/default": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, _mixed_dataset(400), target=1.2, random_seed=1
        ),
        "minlp/approximate-shapley": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, _mixed_dataset(400),
            settings={"approximate_shapley": True, "shapley_samples": 50},
            target=1.2, random_seed=2,
        ),
        "minlp/no-trust-region": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, _mixed_dataset(400),
            settings={"trust_region": False}, target=1.2, random_seed=1,
        ),
        "minlp/no-corrections": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, _mixed_dataset(400),
            settings={"residual_correction": False, "restoration": False, "peak_lock_in": False},
            target=1.2, random_seed=1,
        ),
        "minlp/fixed-feature": lambda: _minlp(
            _mixed_model, mixed(fixed_feature=True), MIXED_SAMPLE, _mixed_dataset(400),
            target=1.0, random_seed=1,
        ),
        "minlp/random-anchor": lambda: _minlp(
            _mixed_model, mixed(), MIXED_SAMPLE, _mixed_dataset(30), target=1.2, random_seed=4
        ),
        "minlp/numerical-only": lambda: _minlp(
            num()[0], num()[1], num()[2], np.random.default_rng(1).uniform(0, 1, (400, 2)),
            target=0.9, random_seed=0,
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


def _snapshot_file(config):
    directory = config.getoption("--snapshot-dir") or SNAPSHOT_DIR
    return Path(directory) / f"{sys.platform}.json"


def _load(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
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
    path = _snapshot_file(request.config)
    if request.config.getoption("--update-snapshots"):
        data = _load(path)
        data["versions"] = {**_versions(), "python": sys.version.split()[0]}
        data["results"][name] = _run(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = dict(sorted(data["results"].items()))
        text = json.dumps({**data, "results": ordered}, indent=1, sort_keys=False)
        path.write_text(text + "\n", encoding="utf-8")
        return

    if not path.exists():
        pytest.skip(f"no snapshots recorded for platform {sys.platform!r}")
    data = _load(path)
    recorded = data["versions"] or {}
    current = _versions()
    if any(recorded.get(lib) != current[lib] for lib in current):
        pytest.skip(f"snapshots recorded with {recorded}, running {current}")
    assert name in data["results"], "no snapshot recorded; run with --update-snapshots"
    _assert_matches(_run(name), data["results"][name])
