import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    return init_triggers(n, True)


@jit("boolean(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> bool:
    """
    Implements Sigma_i (x_i == a) = x_{n-1}.
    :param domains: the domains of the variables
    """
    n = len(domains) - 1
    x = domains[:-1]
    counter = domains[-1]
    value = data[0]
    ko_count = np.count_nonzero((x[:, MIN] > value) | (x[:, MAX] < value))
    ok_count = np.count_nonzero((x[:, MIN] == value) & (x[:, MAX] == value))
    counter[MIN] = max(counter[MIN], ok_count)
    counter[MAX] = min(counter[MAX], n - ko_count)
    if counter[MIN] > counter[MAX]:
        return False
    if ok_count == counter[MAX]:  # we cannot have more domains equal to c
        x[(x[:, MIN] == value) & (x[:, MAX] > value), MIN] = value + 1
        x[(x[:, MIN] < value) & (x[:, MAX] == value), MAX] = value - 1
    if n - ko_count == counter[MIN]:  # we cannot have more domains different from c
        x[(x[:, MIN] <= value) & (value <= x[:, MAX]), :] = value
    return True
