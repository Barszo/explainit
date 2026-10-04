"""Immutable, resumable result storage for the paper studies.

Layout under ``explainit/experiments/paper/results/<study>/``::

    index.csv                  one row per completed run
    runs/<run_id>/spec.json    the exact RunSpec that produced the run
    runs/<run_id>/counterfactuals.csv
    runs/<run_id>/samples.csv
    runs/<run_id>/summary.json

Nothing is ever overwritten unless ``force=True``: a sweep can be interrupted
and resumed, and re-running a study only fills in the missing cells.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from explainit.experiments.paper.common.runspec import RunSpec

PAPER_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PAPER_DIR / "results"


def current_run_ids(study: str):
    """Run ids defined by ``study``'s config today, or ``None`` if unreadable.

    Editing a config leaves earlier runs in the registry; analyses should look
    only at the cells the current configuration asks for.
    """
    from explainit.experiments.paper.common.runspec import (
        expand_grid, load_study_config,
    )

    for config_path in PAPER_DIR.glob("*/config.yaml"):
        try:
            cfg = load_study_config(config_path)
        except Exception:
            continue
        if str(cfg.get("study", config_path.parent.name)) == study:
            return {spec.run_id for spec in expand_grid(cfg)}
    return None

_INDEX_FIELDS = [
    "run_id", "study", "arm", "cell", "dataset", "priority_set", "method",
    "seed", "epsilon", "n_cfs", "n_samples", "n_cfs_valid",
    "avg_priority_score", "avg_priority_score_normalised", "avg_model_rows",
    "avg_time_seconds", "completed_utc", "params",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class RunRegistry:
    """Filesystem registry for one study."""

    def __init__(self, study: str, root: Optional[Path] = None) -> None:
        self.study = study
        self.root = Path(root) if root is not None else RESULTS_ROOT / study
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "index.csv"

    def run_dir(self, spec: RunSpec) -> Path:
        return self.runs_dir / spec.run_id

    def is_complete(self, spec: RunSpec) -> bool:
        return (self.run_dir(spec) / "summary.json").exists()

    def write_run(
        self,
        spec: RunSpec,
        *,
        cf_rows: Sequence[Dict[str, Any]],
        sample_rows: Sequence[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> Path:
        out = self.run_dir(spec)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "spec.json", "w", encoding="utf-8") as handle:
            json.dump(spec.to_json(), handle, indent=2, default=_json_default)
        _write_csv(out / "counterfactuals.csv", cf_rows)
        _write_csv(out / "samples.csv", sample_rows)
        payload = dict(summary)
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        with open(out / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
        self._append_index(spec, payload)
        return out

    def _append_index(self, spec: RunSpec, summary: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        exists = self.index_path.exists()
        row = {
            "run_id": spec.run_id,
            "study": spec.study,
            "arm": spec.arm,
            "cell": spec.cell,
            "dataset": spec.dataset,
            "priority_set": spec.priority_set,
            "method": spec.method,
            "seed": spec.seed,
            "epsilon": spec.epsilon,
            "n_cfs": spec.n_cfs,
            "params": json.dumps(dict(spec.params), sort_keys=True),
        }
        for key in ("n_samples", "n_cfs_valid", "avg_priority_score",
                    "avg_priority_score_normalised", "avg_model_rows",
                    "avg_time_seconds", "completed_utc"):
            row[key] = summary.get(key)
        with open(self.index_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_INDEX_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def load_counterfactuals(self, run_ids=None):
        """All per-CF rows of the study as a DataFrame, annotated with the spec.

        ``run_ids`` restricts the result to a set of ids, which is how callers
        drop runs left over from an earlier version of the study config.
        """
        import pandas as pd

        frames = []
        for spec_path in sorted(self.runs_dir.glob("*/spec.json")):
            cf_path = spec_path.parent / "counterfactuals.csv"
            if not cf_path.exists():
                continue
            with open(spec_path, "r", encoding="utf-8") as handle:
                spec = json.load(handle)
            if run_ids is not None and spec.get("run_id") not in run_ids:
                continue
            frame = pd.read_csv(cf_path)
            for key in ("run_id", "arm", "cell", "dataset", "priority_set",
                        "method", "seed", "epsilon", "n_cfs"):
                frame[key] = spec.get(key)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_index(self):
        import pandas as pd

        if not self.index_path.exists():
            return pd.DataFrame(columns=_INDEX_FIELDS)
        return pd.read_csv(self.index_path)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = ["RESULTS_ROOT", "RunRegistry"]
