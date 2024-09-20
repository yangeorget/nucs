import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_exactly_eq(n: int, data: NDArray) -> float:
    return 2 * n


def get_triggers_exactly_eq(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_exactly_eq(x: NDArray, data: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a_0) = a_1.
    :param domains: the domains of the variables
    :param data: the parameters of the propagator
    """
    value = data[0]
    ok_count_max = len(x) - np.count_nonzero((x[:, MIN] > value) | (x[:, MAX] < value))
    ok_count_min = np.count_nonzero((x[:, MIN] == value) & (x[:, MAX] == value))
    counter = data[1]
    if ok_count_min > counter or ok_count_max < counter:
        return PROP_INCONSISTENCY
    if ok_count_min == counter and ok_count_max == counter:
        return PROP_ENTAILMENT
    if ok_count_min == counter:  # we cannot have more domains equal to c
        x[(x[:, MIN] == value) & (x[:, MAX] > value), MIN] = value + 1
        x[(x[:, MIN] < value) & (x[:, MAX] == value), MAX] = value - 1
    if ok_count_max == counter:  # we cannot have more domains different from c
        x[(x[:, MIN] <= value) & (value <= x[:, MAX]), :] = value
    return PROP_CONSISTENCY
