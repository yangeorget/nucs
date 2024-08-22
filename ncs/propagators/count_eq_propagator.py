import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import (
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    new_triggers,
)


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit("int64(int32[::1,:], int32[:])", cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a) = x_{n-1}.
    :param domains: the domains of the variables
    :param data: the parameters of the propagator
    """
    x = domains[:-1]
    value = data[0]
    ok_count_max = len(x) - np.count_nonzero((x[:, MIN] > value) | (x[:, MAX] < value))
    ok_count_min = np.count_nonzero((x[:, MIN] == value) & (x[:, MAX] == value))
    counter = domains[-1]
    counter[MIN] = max(counter[MIN], ok_count_min)
    counter[MAX] = min(counter[MAX], ok_count_max)
    if counter[MIN] > counter[MAX]:
        return PROP_INCONSISTENCY
    if ok_count_min == ok_count_max:
        return PROP_ENTAILMENT
    if ok_count_min == counter[MAX]:  # we cannot have more domains equal to c
        x[(x[:, MIN] == value) & (x[:, MAX] > value), MIN] = value + 1
        x[(x[:, MIN] < value) & (x[:, MAX] == value), MAX] = value - 1
    if ok_count_max == counter[MIN]:  # we cannot have more domains different from c
        x[(x[:, MIN] <= value) & (value <= x[:, MAX]), :] = value
    return PROP_CONSISTENCY
