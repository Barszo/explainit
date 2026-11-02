"""Random-number helpers shared by the explainers."""

import random
from typing import Any

import numpy as np

# ``numpy.random`` (the module, global stream) or a ``numpy.random.RandomState``;
# both offer the same sampling functions, but no common type describes them.
type NumpyRandom = Any
# ``random`` (the module, global stream) or a ``random.Random``.
type PythonRandom = Any


def make_rngs(seed: int | None = None) -> tuple[NumpyRandom, PythonRandom]:
    """Return ``(numpy_rng, python_rng)`` for one search run.

    With ``seed=None`` the global NumPy and :mod:`random` streams are used, so
    seeding them globally keeps working. With an integer seed, private
    generators are created instead. Their streams are identical to what
    ``np.random.seed(seed)`` / ``random.seed(seed)`` produced in earlier
    versions, so seeded results are unchanged, but the caller's global random
    state is no longer modified.
    """
    if seed is None:
        return np.random, random
    seed = int(seed)
    return np.random.RandomState(seed), random.Random(seed)
