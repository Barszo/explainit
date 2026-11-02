"""Plots of priorities and of how the random search samples them.

* :func:`plot_priorities`: every mutable feature of a priorities dictionary.
* :func:`plot_numerical_priority`, :func:`plot_categorical_priority`: one
  feature or one categorical group.
* :func:`plot_probability_distributions` and the two single-feature versions:
  the sampling distributions of the random search.

Every function writes files only when given ``save_dir`` or ``save_path``,
and shows the plots only with ``show=True``.
"""

from explainit.plotting.categorical import plot_categorical_priority
from explainit.plotting.distributions import (
    plot_categorical_probability_distribution,
    plot_numerical_probability_distribution,
    plot_probability_distributions,
)
from explainit.plotting.numerical import plot_numerical_priority
from explainit.plotting.overview import plot_priorities

__all__ = [
    "plot_categorical_priority",
    "plot_categorical_probability_distribution",
    "plot_numerical_priority",
    "plot_numerical_probability_distribution",
    "plot_priorities",
    "plot_probability_distributions",
]
