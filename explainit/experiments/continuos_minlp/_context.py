"""Internal helper: load an :class:`ExperimentContext` from disk caches.

Loads the pickle written by ``data_setup.py`` and the Keras model written
by ``model_setup.py``, returning a populated ``ExperimentContext``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainit.experiments.continuos_minlp.data_setup import (  # noqa: E402
    load_dataset,
    TARGET_NAMES,
)
from explainit.experiments.continuos_minlp.model_setup import (  # noqa: E402
    load_model,
)
from explainit.experiments.continuos_minlp.priority_sets import (  # noqa: E402
    ExperimentContext,
)


logger = logging.getLogger("explainit.experiments.continuos_minlp._context")


def _as_array(arr):
    return arr.values if hasattr(arr, "values") else np.asarray(arr)


def load_context(dataset_key: str) -> ExperimentContext:
    data: Dict[str, object] = load_dataset(dataset_key)
    model = load_model(dataset_key)
    X_train = _as_array(data["X_train"]).astype(float)
    X_test = _as_array(data["X_test"]).astype(float)
    y_train = np.asarray(data["y_train"], dtype=float).flatten()
    y_test = np.asarray(data["y_test"], dtype=float).flatten()
    feature_names = [str(n) for n in data["feature_names"]]
    target_name = str(data.get("target_name", TARGET_NAMES.get(dataset_key, "target")))
    return ExperimentContext(
        dataset_key=dataset_key,
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_name=target_name,
    )


__all__ = ["load_context"]
