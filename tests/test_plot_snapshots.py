"""Snapshot tests for the plots.

Each scenario draws one plot and records a *fingerprint* of the figure: axes
titles, labels, limits and ticks; every line, patch, collection and text with
its data and style; legend entries; and how many figures are left open. The
fingerprint does not depend on fonts or pixels, so it is stable across
machines, but any change to what is drawn shows up.

Comparisons run only with the NumPy and Matplotlib versions the snapshot was
recorded with (a dedicated CI job pins them). After an intentional change to a
plot, re-record with::

    pytest tests/test_plot_snapshots.py --update-snapshots
"""

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

from explainit.priorities.linear import basic_linear
from explainit.priorities.nonlinear import exponential
from explainit.sampling import sample_numeric_value

try:
    from explainit import plotting as plots
except ImportError:  # layout before the plotting package existed
    from explainit.utils import priority_plots as plots

SNAPSHOT_FILE = Path(__file__).parent / "snapshots" / "figures.json"

pytestmark = pytest.mark.snapshot


def _versions():
    return {
        "numpy": ".".join(np.__version__.split(".")[:2]),
        "matplotlib": ".".join(matplotlib.__version__.split(".")[:2]),
    }


# --- fingerprints -------------------------------------------------------------


def _num(value):
    return float(round(float(value), 10))


def _summary(values):
    """Order-sensitive summary of a numeric array (exact for short arrays)."""
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if arr.size <= 12:
        return [None if not np.isfinite(v) else _num(v) for v in arr]
    weights = np.linspace(1.0, 2.0, finite.size)
    return {
        "n": int(arr.size),
        "nan": int(arr.size - finite.size),
        "first": _num(finite[0]) if finite.size else None,
        "last": _num(finite[-1]) if finite.size else None,
        "min": _num(finite.min()) if finite.size else None,
        "max": _num(finite.max()) if finite.size else None,
        "sum": _num(finite.sum()),
        "weighted": _num(np.dot(finite, weights)),
    }


def _color(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype.kind in "fi" and arr.ndim == 2:
        return [to_hex(c, keep_alpha=True) for c in arr]
    try:
        return to_hex(value, keep_alpha=True)
    except (ValueError, TypeError):
        return str(value)


def _text(t):
    return {
        "text": t.get_text(),
        "position": [_num(v) if isinstance(v, (int, float)) else str(v) for v in t.get_position()],
        "color": _color(t.get_color()),
        "size": _num(t.get_fontsize()),
        "weight": str(t.get_fontweight()),
        "ha": t.get_ha(),
        "va": t.get_va(),
        "rotation": _num(t.get_rotation()),
        "bbox": t.get_bbox_patch() is not None,
        "visible": t.get_visible(),
    }


def _line(line):
    x, y = line.get_data()
    return {
        "label": line.get_label() if not line.get_label().startswith("_") else None,
        "x": _summary(x),
        "y": _summary(y),
        "color": _color(line.get_color()),
        "style": line.get_linestyle(),
        "width": _num(line.get_linewidth()),
        "marker": str(line.get_marker()),
        "size": _num(line.get_markersize()),
        "edge": _color(line.get_markeredgecolor()),
        "edge_width": _num(line.get_markeredgewidth()),
        "alpha": line.get_alpha(),
    }


def _patch(patch):
    entry = {
        "type": type(patch).__name__,
        "face": _color(patch.get_facecolor()),
        "edge": _color(patch.get_edgecolor()),
        "width": _num(patch.get_linewidth()),
        "alpha": patch.get_alpha(),
        "fill": patch.get_fill(),
        "label": patch.get_label() if not patch.get_label().startswith("_") else None,
    }
    if isinstance(patch, Rectangle):
        entry["xywh"] = [_num(patch.get_x()), _num(patch.get_y()), _num(patch.get_width()),
                         _num(patch.get_height())]  # fmt: skip
    else:
        entry["vertices"] = _summary(patch.get_path().vertices)
    return entry


def _collection(collection):
    array = collection.get_array()
    paths = collection.get_paths()
    return {
        "type": type(collection).__name__,
        "offsets": _summary(collection.get_offsets()),
        "array": None if array is None else _summary(array),
        "paths": len(paths),
        "vertices": _summary(np.concatenate([p.vertices for p in paths])) if paths else None,
        "face": _color(collection.get_facecolor()),
        "edge": _color(collection.get_edgecolor()),
        "widths": _summary(collection.get_linewidths()),
        "alpha": collection.get_alpha(),
        "label": collection.get_label() if not collection.get_label().startswith("_") else None,
    }


def _axes(ax):
    legend = ax.get_legend()
    return {
        "title": _text(ax.title),
        "xlabel": _text(ax.xaxis.label),
        "ylabel": _text(ax.yaxis.label),
        "xlim": [_num(v) for v in ax.get_xlim()],
        "ylim": [_num(v) for v in ax.get_ylim()],
        "xticks": _summary(ax.get_xticks()),
        "yticks": _summary(ax.get_yticks()),
        "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
        "tick_params": {
            axis: {k: str(v) for k, v in sorted(getattr(ax, axis).get_tick_params().items())}
            for axis in ("xaxis", "yaxis")
        },
        "facecolor": _color(ax.get_facecolor()),
        "spines": {
            name: [spine.get_visible(), _color(spine.get_edgecolor()), _num(spine.get_linewidth())]
            for name, spine in sorted(ax.spines.items())
        },
        "lines": [_line(line) for line in ax.lines],
        "patches": [_patch(p) for p in ax.patches],
        "collections": [_collection(c) for c in ax.collections],
        "texts": [_text(t) for t in ax.texts],
        "legend": None
        if legend is None
        else {
            "texts": [_text(t) for t in legend.get_texts()],
            "frame": [_color(legend.get_frame().get_facecolor()), legend.get_frame().get_alpha()],
        },
    }


def fingerprint(fig):
    if fig is None:
        return None
    return {
        "size": [_num(v) for v in fig.get_size_inches()],
        "facecolor": _color(fig.get_facecolor()),
        "axes": [_axes(ax) for ax in fig.axes],
    }


# --- scenarios ----------------------------------------------------------------


def _ramp(low, high, increasing=True):
    return {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, low, high, increasing)}


SAMPLE = np.array([0.15, 0.8, 1.0, 0.0, 0.0])
EXEMPLAR = np.array([0.7, 0.35, 0.0, 1.0, 0.0])
PAIRS_SAMPLE = np.array([SAMPLE, [0.4, 0.5, 0.0, 0.0, 1.0]])
PAIRS_EXEMPLAR = np.array([EXEMPLAR, [1.5, 0.9, 1.0, 0.0, 0.0]])  # 1.5 is out of range
GROUP = (2, 3, 4)
WEIGHTS = {(1.0, 0.0, 0.0): 1.0, (0.0, 1.0, 0.0): 0.4, (0.0, 0.0, 1.0): None}


def _data():
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, (300, 5))
    category = rng.integers(0, 3, 300)
    features[:, 2:] = np.eye(3)[category]
    discrete = rng.integers(0, 4, 300) / 3.0
    continuous_target = features[:, 0] + 0.5 * features[:, 1] + 0.1 * rng.normal(size=300)
    class_target = (continuous_target > 0.8).astype(float)
    return features, discrete, continuous_target, class_target


def _numerical(constraint=None, **kwargs):
    return plots.plot_numerical_priority(0, constraint or _ramp(0.2, 0.8), show=False, **kwargs)


def _sampler(seed):
    rng = np.random.RandomState(seed)
    return lambda constraint: sample_numeric_value(constraint, rng=rng)


def scenarios():
    features, discrete, continuous_target, class_target = _data()
    return {
        "numerical/curve": lambda: _numerical(),
        "numerical/immutable": lambda: _numerical({"min": 0.3, "max": 0.3, "function": None}),
        "numerical/sample-exemplar": lambda: _numerical(
            _ramp(0.2, 0.8, increasing=False), sample=SAMPLE, exemplar=EXEMPLAR,
            feature_name="income",
        ),
        "numerical/two-pairs": lambda: _numerical(sample=PAIRS_SAMPLE, exemplar=PAIRS_EXEMPLAR),
        "numerical/continuous-target": lambda: _numerical(
            sample=PAIRS_SAMPLE, exemplar=PAIRS_EXEMPLAR, sample_target_value=0.4,
            exemplar_target_value=[1.1, None], feature_values=features[:, 0],
            target_values=continuous_target, target_name="score", target_threshold=0.9,
        ),
        "numerical/continuous-target-constant-priority": lambda: _numerical(
            {"min": 0.0, "max": 1.0, "function": lambda x: np.ones_like(np.asarray(x, float))},
            sample=SAMPLE, feature_values=features[:, 0], target_values=continuous_target,
        ),
        "numerical/continuous-target-discrete-feature": lambda: _numerical(
            {"min": 0.0, "max": 1.0, "function": lambda x: exponential(x, 0.0, 1.0)},
            sample=np.array([1 / 3, 0, 0, 0, 0]), exemplar=np.array([1.0, 0, 0, 0, 0]),
            feature_values=discrete, target_values=continuous_target, target_threshold=0.9,
        ),
        "numerical/class-target": lambda: _numerical(
            sample=PAIRS_SAMPLE, exemplar=PAIRS_EXEMPLAR, feature_values=features[:, 0],
            target_values=class_target, target_name="approved",
        ),
        "categorical/weights": lambda: plots.plot_categorical_priority(GROUP, WEIGHTS),
        "categorical/sample-exemplar": lambda: plots.plot_categorical_priority(
            GROUP, WEIGHTS, sample=SAMPLE, exemplar=EXEMPLAR,
            feature_names=["income", "savings", "owner", "renter", "other"],
        ),
        "categorical/two-pairs": lambda: plots.plot_categorical_priority(
            GROUP, WEIGHTS, sample=PAIRS_SAMPLE, exemplar=PAIRS_EXEMPLAR
        ),
        "categorical/single-category": lambda: plots.plot_categorical_priority(
            (2,), {(1.0,): 0.7}, sample=SAMPLE
        ),
        "categorical/class-target": lambda: plots.plot_categorical_priority(
            GROUP, WEIGHTS, feature_matrix=features, target_values=class_target,
            target_name="approved",
        ),
        "categorical/continuous-target": lambda: plots.plot_categorical_priority(
            GROUP, WEIGHTS, feature_matrix=features, target_values=continuous_target,
        ),
        "distribution/numerical": lambda: plots.plot_numerical_probability_distribution(
            0, _ramp(0.2, 0.8), n_samples=500, sampler=_sampler(0), feature_name="income"
        ),
        "distribution/categorical": lambda: plots.plot_categorical_probability_distribution(
            GROUP, WEIGHTS, feature_names=["a", "b", "c", "d", "e"]
        ),
        "distribution/categorical-single": lambda: plots.plot_categorical_probability_distribution(
            (2,), {(1.0,): 0.7}
        ),
        "distribution/categorical-forbidden": lambda: (
            plots.plot_categorical_probability_distribution(GROUP, {(1.0, 0.0, 0.0): 0})
        ),
    }  # fmt: skip


SCENARIOS = scenarios()


def _run(name):
    plt.close("all")
    logging.disable(logging.CRITICAL)
    try:
        with plt.rc_context():
            plt.rcdefaults()
            fig = SCENARIOS[name]()
            return {"figure": fingerprint(fig), "open_figures": len(plt.get_fignums())}
    finally:
        logging.disable(logging.NOTSET)
        plt.close("all")


def _load():
    if SNAPSHOT_FILE.exists():
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    return {"versions": None, "results": {}}


def _assert_matches(actual, expected, path="figure"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys differ"
        for key in expected:
            _assert_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: length differs"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_matches(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) and isinstance(actual, float):
        assert actual == pytest.approx(expected, rel=1e-7, abs=1e-9), path
    else:
        assert actual == expected, path


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_plot_snapshot(name, request):
    if request.config.getoption("--update-snapshots"):
        data = _load()
        data["versions"] = _versions()
        data["results"][name] = json.loads(json.dumps(_run(name)))
        SNAPSHOT_FILE.parent.mkdir(exist_ok=True)
        ordered = dict(sorted(data["results"].items()))
        SNAPSHOT_FILE.write_text(
            json.dumps({**data, "results": ordered}, indent=1) + "\n", encoding="utf-8"
        )
        return

    data = _load()
    if data["versions"] != _versions():
        pytest.skip(f"figures recorded with {data['versions']}, running {_versions()}")
    assert name in data["results"], "no snapshot recorded; run with --update-snapshots"
    _assert_matches(json.loads(json.dumps(_run(name))), data["results"][name])
