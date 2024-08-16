import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    triggers = init_triggers(n, False)
    for i in range(n-1):
        triggers[i, MIN] = True
    triggers[-1, MAX] = True
    return triggers


@jit("boolean(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> bool:
    """
    Implements Max_i x_i = x_{n-1}.
    :param domains: the domains of the variables
    """
    n = len(domains) - 1
    x = domains[:-1]
    y = domains[-1]
    y[MIN] = max(y[MIN], np.max(x[:, MIN]))
    if y[MIN] > y[MAX]:
        return False
    for i in range(n):
        x[i, MAX] = min(x[i, MAX], y[MAX])
        if x[i, MAX] < x[i, MIN]:
            return False
    return True
