import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = init_triggers(n, False)
    for i in range(n - 1):
        triggers[i, MAX] = True
    triggers[-1, MIN] = True
    return triggers


@jit("boolean(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> bool:
    """
    Implements Min_i x_i = x_{n-1}.
    :param domains: the domains of the variables
    """
    x = domains[:-1]
    y = domains[-1]
    y[MAX] = min(y[MAX], np.min(x[:, MAX]))
    if y[MIN] > y[MAX]:
        return False
    for i in range(len(x)):
        x[i, MIN] = max(x[i, MIN], y[MIN])
        if x[i, MAX] < x[i, MIN]:
            return False
    return True
