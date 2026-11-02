import numpy as np
import numpy.typing as npt

################################
# Exponential functions
#################################


def exponential(
    x: npt.ArrayLike, x0: float, x1: float, increasing: bool = True, a: float = 5
) -> npt.NDArray[np.float64]:
    """
    Exponential function from 0 to 1 (increasing) or 1 to 0 (decreasing).
    If increasing=True: 0 for x <= x0, 1 for x >= x1.
    If increasing=False: 1 for x <= x0, 0 for x >= x1.
    a : controls the steepness of the transition (the greater value of a increases steepness)
    Always returns a numpy array.
    """
    x = np.asarray(x)
    if x1 > x0:
        t = (x - x0) / (x1 - x0)
        t = np.clip(t, 0, 1)
        curve = (np.exp(a * t) - 1) / (np.exp(a) - 1)
        if increasing:
            val = np.where(x <= x0, 0.0, np.where(x >= x1, 1.0, curve))
        else:
            val = np.where(x <= x0, 1.0, np.where(x >= x1, 0.0, 1 - curve))
    else:
        val = np.zeros_like(x) if increasing else np.ones_like(x)
    return val


def basic_linear_step(
    x: npt.ArrayLike, x0: float, x1: float, increasing: bool = True
) -> npt.NDArray[np.float64]:
    """
    Linear function from 0 to 1 (increasing) or 1 to 0 (decreasing).
    If increasing=True: 0 for x <= x0, 1 for x >= x1.
    If increasing=False: 1 for x <= x0, 0 for x >= x1.
    Always returns a numpy array.
    """
    x = np.asarray(x)
    if x1 > x0:
        t = (x - x0) / (x1 - x0)
        t = np.clip(t, 0, 1)
        if increasing:
            val = np.where(x <= x0, 0.0, np.where(x >= x1, 1.0, t))
        else:
            val = np.where(x <= x0, 1.0, np.where(x >= x1, 0.0, 1 - t))
    else:
        val = np.zeros_like(x) if increasing else np.ones_like(x)
    return val
