import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return init_triggers(n, True)


@jit("boolean(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(x: NDArray, data: NDArray) -> bool:
    """
    Implements Sigma_i (x_i == a_0) = a_1.
    :param domains: the domains of the variables
    """
    value = data[0]
    ok_count_max = len(x) - np.count_nonzero((x[:, MIN] > value) | (x[:, MAX] < value))
    ok_count_min = np.count_nonzero((x[:, MIN] == value) & (x[:, MAX] == value))
    counter = data[1]
    if ok_count_min > counter or ok_count_max < counter:
        return False
    if ok_count_min == counter:  # we cannot have more domains equal to c
        x[(x[:, MIN] == value) & (x[:, MAX] > value), MIN] = value + 1
        x[(x[:, MIN] < value) & (x[:, MAX] == value), MAX] = value - 1
    if ok_count_max == counter:  # we cannot have more domains different from c
        x[(x[:, MIN] <= value) & (value <= x[:, MAX]), :] = value
    return True
