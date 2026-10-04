"""Run a study: expand its grid, execute the missing cells, store the results.

Usage::

    python -m explainit.experiments.paper.run_study \
        --config explainit/experiments/paper/study1_hyperparams/config.yaml --dry-run
    python -m explainit.experiments.paper.run_study \
        --config explainit/experiments/paper/study1_hyperparams/config.yaml
    # resume a sweep / add newly added cells only:
    python -m explainit.experiments.paper.run_study --config <cfg>          # skips completed
    python -m explainit.experiments.paper.run_study --config <cfg> --force  # re-runs everything

Runs are stored one directory per cell under
``explainit/experiments/paper/results/<study>/runs/<run_id>/`` and indexed in
``index.csv``; nothing is overwritten unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.paper.common.execute import execute_run  # noqa: E402
from explainit.experiments.paper.common.registry import RunRegistry  # noqa: E402
from explainit.experiments.paper.common.runspec import (  # noqa: E402
    RunSpec,
    describe_grid,
    expand_grid,
    load_study_config,
)

logger = logging.getLogger("explainit.experiments.paper.run_study")


def _filter(specs: Sequence[RunSpec], args: argparse.Namespace) -> List[RunSpec]:
    out = list(specs)
    if args.dataset:
        out = [s for s in out if s.dataset in args.dataset]
    if args.arm:
        out = [s for s in out if s.arm in args.arm]
    if args.method:
        out = [s for s in out if s.method in args.method]
    if args.limit:
        out = out[: int(args.limit)]
    return out


def run_study(config_path: Path, args: argparse.Namespace) -> None:
    cfg = load_study_config(config_path)
    specs = _filter(expand_grid(cfg), args)
    study = str(cfg.get("study", config_path.parent.name))
    registry = RunRegistry(study, root=Path(args.results_root) / study
                           if args.results_root else None)

    logger.info("Study '%s': %d run(s) after filtering.", study, len(specs))
    if args.dry_run:
        print(describe_grid(specs))
        return

    started = time.perf_counter()
    executed = skipped = failed = 0
    for position, spec in enumerate(specs, start=1):
        if not args.force and registry.is_complete(spec):
            skipped += 1
            logger.info("[%d/%d] skip (already complete): %s",
                        position, len(specs), spec.run_id)
            continue
        logger.info("[%d/%d] running %s | arm=%s cell=%s",
                    position, len(specs), spec.run_id, spec.arm, spec.cell)
        try:
            cf_rows, sample_rows, summary = execute_run(spec)
        except Exception as exc:
            failed += 1
            logger.error("[%d/%d] run failed (%s): %s",
                         position, len(specs), spec.run_id, exc)
            continue
        registry.write_run(spec, cf_rows=cf_rows, sample_rows=sample_rows,
                           summary=summary)
        executed += 1
        elapsed = time.perf_counter() - started
        eta = (elapsed / executed) * (len(specs) - position) if executed else 0.0
        logger.info(
            "[%d/%d] done: valid=%d/%d | avg_priority_norm=%s | avg_model_rows=%s "
            "| eta=%.0fs",
            position, len(specs), summary["n_cfs_valid"], summary["n_cfs_total"],
            summary["avg_priority_score_normalised"], summary["avg_model_rows"], eta,
        )

    logger.info("Study '%s' finished: %d executed, %d skipped, %d failed in %.0fs. "
                "Index: %s", study, executed, skipped, failed,
                time.perf_counter() - started, registry.index_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to a study config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the expanded grid and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run cells that already have results.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Execute at most N runs (smoke tests).")
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--arm", nargs="*", default=None)
    parser.add_argument("--method", nargs="*", default=None)
    parser.add_argument("--results-root", default=None,
                        help="Override the results root (default: paper/results).")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    # The explainer logs every stage at INFO; keep study logs readable unless -v.
    logging.getLogger("explainit").setLevel(
        logging.DEBUG if args.verbose else logging.WARNING)
    for name in ("explainit.experiments.paper.run_study",
                 "explainit.experiments.paper.execute"):
        logging.getLogger(name).setLevel(logging.INFO)
    run_study(Path(args.config), args)


if __name__ == "__main__":
    main()
