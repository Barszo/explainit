"""Quality benchmark for the explainit search methods.

Runs every method on small real-data tasks and reports how often it finds a
valid counterfactual, how preferable the answers are, and how much model work
and time it needs. Results are written as JSON so changes can be compared with
``results/baseline.json``.

Usage (from the repository root)::

    uv run --group benchmark python benchmarks/run_benchmark.py
    uv run --group benchmark python benchmarks/run_benchmark.py --output benchmarks/results/new.json
    uv run --group benchmark python benchmarks/run_benchmark.py --compare benchmarks/results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import explainit
from explainit import MINLPSearchExplainer, RandomSearchExplainer, basic_linear, exponential

EPSILON = 0.05
N_INSTANCES = 8


class CountingModel:
    def __init__(self, fn):
        self.fn = fn
        self.rows = 0

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        self.rows += X.shape[0]
        return self.fn(X)


@dataclass
class Task:
    name: str
    kind: str  # "regression" or "binary"
    predict: object
    priorities: dict
    fixed: dict
    instances: np.ndarray
    targets: np.ndarray
    dataset: np.ndarray

    def max_score(self):
        return sum(1.0 for c in self.priorities["numerical"].values() if c["function"])

    def score(self, x):
        return float(
            sum(
                float(c["function"](x[i]))
                for i, c in self.priorities["numerical"].items()
                if c["function"] is not None
            )
        )

    def respects_constraints(self, x):
        for i, c in self.priorities["numerical"].items():
            if c["function"] is None:
                if not np.isclose(x[i], self.fixed[i]):
                    return False
            elif not c["min"] - 1e-9 <= x[i] <= c["max"] + 1e-9:
                return False
        return True

    def reaches_target(self, x, target):
        pred = float(self.predict(np.asarray([x], dtype=float))[0])
        if self.kind == "binary":
            return pred > 0.5
        return abs(pred - target) <= EPSILON


def _priorities(X, fixed_columns, seed):
    """Ranges from the 5th-95th percentile; a ramp per feature in a seeded direction."""
    rng = np.random.default_rng(seed)
    numerical = {}
    for i in range(X.shape[1]):
        if i in fixed_columns:
            numerical[i] = {"min": None, "max": None, "function": None}
            continue
        low, high = np.percentile(X[:, i], [5, 95])
        increasing = bool(rng.random() < 0.5)
        shape = exponential if rng.random() < 0.5 else basic_linear
        numerical[i] = {
            "min": float(low),
            "max": float(high),
            "function": lambda x, a=float(low), b=float(high), inc=increasing, f=shape: f(
                x, a, b, increasing=inc
            ),
        }
    return {"numerical": numerical, "categorical": {}}


def _with_fixed(priorities, x):
    """Copy of ``priorities`` with immutable features pinned to ``x``."""
    numerical, fixed = {}, {}
    for i, cfg in priorities["numerical"].items():
        if cfg["function"] is None:
            fixed[i] = float(x[i])
            numerical[i] = {"min": float(x[i]), "max": float(x[i]), "function": None}
        else:
            numerical[i] = dict(cfg)
    return {"numerical": numerical, "categorical": {}}, fixed


def diabetes_task():
    data = load_diabetes()
    X = StandardScaler().fit_transform(data.data)
    y = (data.target - data.target.mean()) / data.target.std()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Explain patients with a low predicted progression; ask for +0.75 std.
    rows = X_test[np.argsort(predictions)[:N_INSTANCES]]
    targets = model.predict(rows) + 0.75
    # age (0) and sex (1) cannot change.
    priorities = _priorities(X_train, fixed_columns={0, 1}, seed=0)
    return Task("diabetes/gbr/regression", "regression", model.predict, priorities, {},
                rows, targets, X_train)  # fmt: skip


def breast_cancer_task():
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data[:, :6])
    y = data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    def predict(X):
        return model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    probabilities = predict(X_test)
    rows = X_test[probabilities < 0.5][:N_INSTANCES]
    priorities = _priorities(X_train, fixed_columns={1}, seed=1)
    return Task("breast_cancer/logreg/binary", "binary", predict, priorities, {},
                rows, np.ones(len(rows)), X_train)  # fmt: skip


def run_random(task, x, target, priorities):
    model = CountingModel(task.predict)
    explainer = RandomSearchExplainer(model, priorities, x, target)
    common = {"expected_counterfactuals": 1, "max_iterations": 5000, "random_seed": 0,
              "n_candidates_per_cf": 50}  # fmt: skip
    if task.kind == "binary":
        cfs, *_ = explainer.generate_for_binary(target_class=1, threshold=0.5, **common)
    else:
        cfs, *_ = explainer.generate_random_samples(epsilon=EPSILON, **common)
    return (np.asarray(cfs[0], dtype=float) if cfs else None), model.rows


def run_minlp(task, x, target, priorities):
    model = CountingModel(task.predict)
    explainer = MINLPSearchExplainer(model, priorities, x, target, task.dataset, epsilon=EPSILON)
    if task.kind == "binary":
        cfs, _, scores, _ = explainer.find_counterfactuals_for_binary(target_class=1)
        cf = np.asarray(cfs[int(np.argmax(scores))], dtype=float) if cfs else None
    else:
        result = explainer.find_counterfactuals(random_seed=0)
        cf = None if result is None else np.asarray(result, dtype=float)
    return cf, model.rows


METHODS = {"random_search": run_random, "minlp_search": run_minlp}


def evaluate(task, method):
    records = []
    for x, target in zip(task.instances, task.targets, strict=True):
        priorities, task.fixed = _with_fixed(task.priorities, x)
        start = time.perf_counter()
        try:
            cf, rows = METHODS[method](task, x.copy(), float(target), priorities)
            error = None
        except Exception as exc:  # benchmark records failures instead of stopping
            cf, rows, error = None, None, f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start
        found = cf is not None
        reached = found and task.reaches_target(cf, target)
        respects = found and task.respects_constraints(cf)
        records.append({
            "found": found,
            "reached_target": bool(reached),
            "respects_constraints": bool(respects),
            "valid": bool(reached and respects),
            "relative_score": task.score(cf) / task.max_score() if found else None,
            "model_rows": rows,
            "seconds": seconds,
            "error": error,
        })  # fmt: skip
    return records


def summarise(records):
    def mean(key, subset=None):
        values = [r[key] for r in (subset or records) if r[key] is not None]
        return float(np.mean(values)) if values else None

    valid = [r for r in records if r["valid"]]
    return {
        "instances": len(records),
        "valid_rate": len(valid) / len(records),
        "reached_target_rate": float(np.mean([r["reached_target"] for r in records])),
        "respects_constraints_rate": float(np.mean([r["respects_constraints"] for r in records])),
        "mean_relative_score_of_valid": mean("relative_score", valid),
        "mean_model_rows": mean("model_rows"),
        "mean_seconds": mean("seconds"),
        "errors": sum(r["error"] is not None for r in records),
    }


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run():
    logging.disable(logging.CRITICAL)
    results = {}
    for task in (diabetes_task(), breast_cancer_task()):
        for method in METHODS:
            records = evaluate(task, method)
            results[f"{task.name}/{method}"] = {"summary": summarise(records), "runs": records}
    return {
        "metadata": {
            "explainit": explainit.__version__,
            "commit": _git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "results": results,
    }


def format_table(report, baseline=None):
    columns = [
        ("valid_rate", "valid", "{:.0%}"),
        ("mean_relative_score_of_valid", "score", "{:.3f}"),
        ("mean_model_rows", "model rows", "{:,.0f}"),
        ("mean_seconds", "seconds", "{:.2f}"),
        ("errors", "errors", "{}"),
    ]
    lines = ["| task / method | " + " | ".join(c[1] for c in columns) + " |",
             "| --- |" + " ---: |" * len(columns)]  # fmt: skip
    for name, entry in report["results"].items():
        cells = []
        for key, _, fmt in columns:
            value = entry["summary"][key]
            cell = "n/a" if value is None else fmt.format(value)
            old = (baseline or {}).get("results", {}).get(name, {}).get("summary", {}).get(key)
            if baseline is not None and old is not None and value is not None:
                cell += f" ({fmt.format(old)})"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument("--compare", type=Path, help="show baseline values in parentheses")
    args = parser.parse_args(argv)

    report = run()
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    print(format_table(report, baseline))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
