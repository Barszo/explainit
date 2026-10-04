"""Is anything still running, and how far along is it?

Scans ``explainit/experiments/paper/study_*/config.yaml`` for the expected
number of runs, counts what the registry has actually completed, and reports
progress, throughput and an ETA per study. Also says whether a ``run_study``
process is currently alive.

Usage::

    python -m explainit.experiments.paper.status
    python -m explainit.experiments.paper.status --watch      # refresh until done
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.paper.common.registry import RESULTS_ROOT  # noqa: E402
from explainit.experiments.paper.common.runspec import (  # noqa: E402
    expand_grid,
    load_study_config,
)

PAPER_DIR = Path(__file__).resolve().parent


def _running_processes() -> List[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "explainit.experiments.paper.run_study"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    labels = []
    for line in lines:
        pid = line.split()[0]
        config = ""
        for token in line.split():
            if token.endswith(".yaml"):
                config = Path(token).parent.name or token
        # A wrapper that polls before launching is queued, not executing.
        state = "queued " if ("pgrep" in line or "while" in line) else "running"
        labels.append(f"{state} pid {pid} ({config or 'unknown config'})")
    return labels


def _expected_runs() -> Dict[str, int]:
    expected: Dict[str, int] = {}
    for config_path in sorted(PAPER_DIR.glob("*/config.yaml")):
        try:
            cfg = load_study_config(config_path)
            expected[str(cfg.get("study", config_path.parent.name))] = len(expand_grid(cfg))
        except Exception:
            continue
    return expected


def _expected_run_ids() -> Dict[str, set]:
    """Run ids the current configs ask for, so stale runs are not counted as done."""
    ids: Dict[str, set] = {}
    for config_path in sorted(PAPER_DIR.glob("*/config.yaml")):
        try:
            cfg = load_study_config(config_path)
            ids[str(cfg.get("study", config_path.parent.name))] = {
                spec.run_id for spec in expand_grid(cfg)}
        except Exception:
            continue
    return ids


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "-"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def report(studies: Optional[Sequence[str]] = None) -> bool:
    """Print one line per study; returns True when everything is finished."""
    import pandas as pd

    expected = _expected_runs()
    expected_ids = _expected_run_ids()
    processes = _running_processes()
    roots = sorted(p for p in RESULTS_ROOT.glob("*") if p.is_dir()) if RESULTS_ROOT.exists() else []
    names = [p.name for p in roots]
    for name in expected:
        if name not in names:
            names.append(name)
    if studies:
        names = [n for n in names if n in studies]

    print(f"\n{'study':28s} {'done':>12s} {'progress':>9s} {'last completed (UTC)':>21s} "
          f"{'per run':>9s} {'eta':>8s}")
    print("-" * 92)
    incomplete: List[str] = []
    for name in sorted(names):
        index_path = RESULTS_ROOT / name / "index.csv"
        total = expected.get(name)
        done = 0
        stale = 0
        last = "-"
        per_run = None
        eta = None
        if index_path.exists():
            frame = pd.read_csv(index_path)
            wanted = expected_ids.get(name)
            if wanted is not None and "run_id" in frame.columns:
                stale = int((~frame["run_id"].isin(wanted)).sum())
                frame = frame[frame["run_id"].isin(wanted)]
            else:
                stale = 0
            done = len(frame)
            if done and "completed_utc" in frame.columns:
                stamps = pd.to_datetime(frame["completed_utc"], errors="coerce", utc=True).dropna()
                if len(stamps):
                    last = stamps.max().strftime("%Y-%m-%d %H:%M:%S")
                if len(stamps) >= 2:
                    span = (stamps.max() - stamps.min()).total_seconds()
                    per_run = span / max(1, len(stamps) - 1)
                    if total:
                        eta = per_run * max(0, total - done)
        if total and done < total:
            incomplete.append(name)
        pct = f"{100.0 * done / total:.0f}%" if total else "-"
        total_str = f"{done}/{total}" if total else str(done)
        print(f"{name:28s} {total_str:>12s} {pct:>9s} {last:>21s} "
              f"{_format_eta(per_run):>9s} {_format_eta(eta):>8s}"
              + (f"  (+{stale} stale)" if stale else ""))

    print()
    if processes:
        print("RUNNING: " + "; ".join(processes))
    else:
        print("RUNNING: nothing -- all studies have stopped.")
    if not processes and incomplete:
        print(f"INCOMPLETE (re-run the same command to resume): {', '.join(incomplete)}")
    return not processes


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", nargs="*", default=None)
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every --interval seconds until nothing runs.")
    parser.add_argument("--interval", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    while True:
        finished = report(args.study)
        if finished or not args.watch:
            break
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
