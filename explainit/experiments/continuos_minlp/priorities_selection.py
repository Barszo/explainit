"""Stage 4: priorities selection workbench (edit-and-run).

Pick a dataset, a priority set name from ``priority_sets.py``, one or more
test-set sample indices and one or more target prediction values, then
run this script to get:

* per-feature coverage report (``analysis/<dataset>/<sample>_<target>/coverage.txt``),
* dataset analysis plots,
* priority surface plots with every sample/exemplar pair overlaid,
* a list of closest exemplars per target.

All outputs land under
``analysis/<dataset_key>/<sample_idx>_<target>/`` plus a ``combined/``
folder when multiple (sample, target) pairs are requested.

Workflow: edit the ``USER_*`` constants near the top, run the file, look
at the plots, tweak the priority builder in ``priority_sets.py``, run
again.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.utils.priorities_analyser import analyse_priorities  # noqa: E402
from explainit.experiments.continuos_minlp._context import load_context  # noqa: E402
from explainit.experiments.continuos_minlp.priority_sets import (  # noqa: E402
    ExperimentContext,
    get_priority_set,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"

logger = logging.getLogger("explainit.experiments.continuos_minlp.priorities_selection")


# ---------------------------------------------------------------------------
# User constants -- edit these
# ---------------------------------------------------------------------------


USER_DATASET: str = "diabetes"
USER_PRIORITY_SET: str = "default"
USER_SAMPLE_INDICES: Sequence[int] = (0, 5)
USER_TARGETS_SCALED: Sequence[float] = (0.25, 0.75)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_target(value: float) -> str:
    s = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _analysis_dir_for(dataset_key: str, sample_idx: int, target_y: float) -> Path:
    sub = f"{int(sample_idx)}_{_format_target(target_y)}"
    return ANALYSIS_DIR / dataset_key / sub


def _build_priorities_for(
    ctx: ExperimentContext,
    sample: np.ndarray,
    target_y: float,
    priority_set: str,
):
    builder = get_priority_set(ctx.dataset_key, priority_set)
    return builder(ctx, sample, float(target_y))


def run_selection(
    ctx: ExperimentContext,
    sample_indices: Sequence[int],
    targets: Sequence[float],
    priority_set: str,
) -> List[Path]:
    """Run the full analyser for every (sample, target) combination.

    For each pair we generate the priorities (so coverage reflects the
    specific pair), then collect the per-pair sample/exemplar vectors so
    the combined folder shows all of them overlaid.
    """

    written: List[Path] = []
    all_samples: List[np.ndarray] = []
    all_exemplars: List[np.ndarray] = []
    all_targets: List[float] = []
    combined_priorities = None

    for sample_idx in sample_indices:
        if not (0 <= int(sample_idx) < len(ctx.X_test)):
            logger.warning(
                "Sample index %s out of range (test size %d); skipping.",
                sample_idx, len(ctx.X_test),
            )
            continue
        sample = ctx.X_test[int(sample_idx)].astype(float)

        for target_y in targets:
            logger.info(
                "Running analyser for dataset=%s sample=%d target=%.4f priority_set=%s",
                ctx.dataset_key, int(sample_idx), float(target_y), priority_set,
            )
            priorities = _build_priorities_for(
                ctx, sample, float(target_y), priority_set,
            )
            if combined_priorities is None:
                combined_priorities = priorities

            out_dir = _analysis_dir_for(ctx.dataset_key, int(sample_idx), float(target_y))
            report = analyse_priorities(
                model=ctx.model,
                dataset=ctx.X_train,
                priorities=priorities,
                feature_names=ctx.feature_names,
                target_values=[float(target_y)],
                samples=[sample],
                y_full=ctx.y_train,
                target_name=ctx.target_name,
                dataset_key=ctx.dataset_key,
                output_dir=out_dir,
            )
            written.extend(report.saved_plots)
            if report.coverage_text_path is not None:
                written.append(report.coverage_text_path)
            if report.summary_text_path is not None:
                written.append(report.summary_text_path)

            all_samples.append(sample)
            if report.exemplars:
                all_exemplars.append(report.exemplars[0].vector)
                all_targets.append(report.exemplars[0].target_value)

    if len(all_samples) > 1 and combined_priorities is not None:
        combined_dir = ANALYSIS_DIR / ctx.dataset_key / "combined"
        logger.info("Rendering combined view in %s", combined_dir)
        report = analyse_priorities(
            model=ctx.model,
            dataset=ctx.X_train,
            priorities=combined_priorities,
            feature_names=ctx.feature_names,
            target_values=all_targets,
            samples=all_samples,
            exemplars=all_exemplars,
            y_full=ctx.y_train,
            target_name=ctx.target_name,
            dataset_key=ctx.dataset_key,
            output_dir=combined_dir,
        )
        written.extend(report.saved_plots)
        if report.coverage_text_path is not None:
            written.append(report.coverage_text_path)
        if report.summary_text_path is not None:
            written.append(report.summary_text_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", "-d", default=USER_DATASET,
        help="Dataset key (must be registered in priority_sets.py).",
    )
    parser.add_argument(
        "--priority-set", default=USER_PRIORITY_SET,
        help="Priority set name within the chosen dataset.",
    )
    parser.add_argument(
        "--sample", type=int, action="append", default=None,
        help="Test-set sample index (repeatable). Overrides USER_SAMPLE_INDICES.",
    )
    parser.add_argument(
        "--target", type=float, action="append", default=None,
        help="Scaled target prediction value (repeatable). Overrides USER_TARGETS_SCALED.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    sample_indices = list(args.sample) if args.sample else list(USER_SAMPLE_INDICES)
    targets = list(args.target) if args.target else list(USER_TARGETS_SCALED)

    ctx = load_context(args.dataset)
    logger.info(
        "Loaded context for '%s' | %d features | train=%d test=%d",
        ctx.dataset_key, len(ctx.feature_names),
        len(ctx.X_train), len(ctx.X_test),
    )
    written = run_selection(ctx, sample_indices, targets, args.priority_set)
    logger.info("Wrote %d artefact(s) under %s", len(written), ANALYSIS_DIR / ctx.dataset_key)


if __name__ == "__main__":
    main()
