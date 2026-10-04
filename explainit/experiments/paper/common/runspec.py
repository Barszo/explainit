"""Expand a study configuration into immutable, hashed run specifications.

A study config is a YAML file of the form::

    study: study1_hyperparams

    defaults:
      epsilon: 0.05
      n_cfs: 1
      seeds: [0, 1, 2]
      selection:
        strategy: indices
        sample_indices: [60, 2, 52]
        target_offset: -0.3
        skip_if_target_below: 0.0

    arms:
      - name: minlp_shap_budget
        method: minlp
        dataset: [diabetes]
        priority_set: [set1, set2]
        n_cfs: 1
        params:
          shap_approx: true              # scalar -> constant
          shap_num_samples: [50, 200]    # list   -> swept
          max_iterations: 10

Every list-valued entry under ``dataset``, ``priority_set`` and ``params`` is
swept; scalars are held constant. The cartesian product of the swept entries
and the seeds gives one :class:`RunSpec` per cell.

To sweep a parameter whose value *is* a list, wrap it in an outer list
(``[[1, 2], [3, 4]]``).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class RunSpec:
    """One executable cell of a study: one method, one config, one seed."""

    study: str
    arm: str
    dataset: str
    priority_set: str
    method: str
    params: Mapping[str, Any]
    selection: Mapping[str, Any]
    epsilon: float
    n_cfs: int
    seed: int
    #: params that vary inside this arm, used for plot/table labels
    swept_keys: Tuple[str, ...] = field(default=())

    @property
    def run_id(self) -> str:
        digest = hashlib.sha1(_canonical(self.identity).encode("utf-8")).hexdigest()[:10]
        return f"{self.dataset}__{self.priority_set}__{self.method}__s{self.seed}__{digest}"

    @property
    def identity(self) -> Dict[str, Any]:
        """Everything that makes this run distinct (drives ``run_id``)."""
        return {
            "study": self.study,
            "arm": self.arm,
            "dataset": self.dataset,
            "priority_set": self.priority_set,
            "method": self.method,
            "params": dict(self.params),
            "selection": dict(self.selection),
            "epsilon": self.epsilon,
            "n_cfs": self.n_cfs,
            "seed": self.seed,
        }

    @property
    def cell(self) -> str:
        """Human-readable label of the swept coordinates, e.g. ``shap_num_samples=50``."""
        if not self.swept_keys:
            return self.arm
        parts = [f"{k}={self.params[k]}" for k in self.swept_keys if k in self.params]
        return ",".join(parts) if parts else self.arm

    def to_json(self) -> Dict[str, Any]:
        payload = dict(self.identity)
        payload["run_id"] = self.run_id
        payload["cell"] = self.cell
        payload["swept_keys"] = list(self.swept_keys)
        return payload


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else [value]


def _expand_params(params: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Tuple[str, ...]]:
    keys = list(params)
    options = [_as_list(params[k]) for k in keys]
    swept = tuple(k for k, opts in zip(keys, options) if len(opts) > 1)
    combos = [dict(zip(keys, values)) for values in itertools.product(*options)] or [{}]
    return combos, swept


def load_study_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if "arms" not in cfg:
        raise ValueError(f"Study config {path} has no 'arms' section.")
    cfg.setdefault("study", path.parent.name)
    return cfg


def expand_grid(cfg: Mapping[str, Any]) -> List[RunSpec]:
    """Expand a loaded study config into one :class:`RunSpec` per cell."""

    study = str(cfg.get("study", "study"))
    defaults = dict(cfg.get("defaults", {}) or {})
    default_selection = dict(defaults.get("selection", {}) or {})
    default_seeds = [int(s) for s in _as_list(defaults.get("seeds", [0]))]

    specs: List[RunSpec] = []
    for arm in cfg.get("arms", []) or []:
        name = str(arm.get("name") or arm["method"])
        method = str(arm["method"])
        epsilon = float(arm.get("epsilon", defaults.get("epsilon", 0.05)))
        n_cfs = int(arm.get("n_cfs", defaults.get("n_cfs", 1)))
        selection = {**default_selection, **dict(arm.get("selection", {}) or {})}
        seeds = [int(s) for s in _as_list(arm.get("seeds", default_seeds))]
        datasets = [str(d) for d in _as_list(arm.get("dataset", defaults.get("dataset")))]
        priority_sets = [
            str(p) for p in _as_list(arm.get("priority_set", defaults.get("priority_set")))
        ]
        param_combos, swept = _expand_params(dict(arm.get("params", {}) or {}))

        for dataset, priority_set, params, seed in itertools.product(
            datasets, priority_sets, param_combos, seeds
        ):
            specs.append(
                RunSpec(
                    study=study,
                    arm=name,
                    dataset=dataset,
                    priority_set=priority_set,
                    method=method,
                    params=dict(params),
                    selection=dict(selection),
                    epsilon=epsilon,
                    n_cfs=n_cfs,
                    seed=seed,
                    swept_keys=swept,
                )
            )

    duplicates = _duplicate_ids(specs)
    if duplicates:
        raise ValueError(f"Study grid produced duplicate run ids: {duplicates}")
    return specs


def _duplicate_ids(specs: Iterable[RunSpec]) -> List[str]:
    seen: Dict[str, int] = {}
    for spec in specs:
        seen[spec.run_id] = seen.get(spec.run_id, 0) + 1
    return [rid for rid, count in seen.items() if count > 1]


def describe_grid(specs: Sequence[RunSpec]) -> str:
    lines = [f"{len(specs)} run(s):"]
    for spec in specs:
        lines.append(
            f"  {spec.run_id}  arm={spec.arm}  cell={spec.cell}  "
            f"n_cfs={spec.n_cfs}  eps={spec.epsilon}"
        )
    return "\n".join(lines)


__all__ = ["RunSpec", "describe_grid", "expand_grid", "load_study_config"]
