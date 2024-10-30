###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024 - Yan Georget
###############################################################################
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_count_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 2 * n


def get_triggers_count_eq(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.ones((n, 2), dtype=np.bool)


@njit(cache=True)
def compute_domains_count_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a) = x_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is the first parameter
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    a = parameters[0]
    x = domains[:-1]
    counter = domains[-1]
    count_max = len(x)
    count_min = 0
    for x_i in x:
        if x_i[MIN] > a or x_i[MAX] < a:
            count_max -= 1
        elif x_i[MIN] == a and x_i[MAX] == a:
            count_min += 1
    counter[MIN] = max(counter[MIN], count_min)
    counter[MAX] = min(counter[MAX], count_max)
    if counter[MIN] > counter[MAX]:
        return PROP_INCONSISTENCY
    if count_min == count_max:
        return PROP_ENTAILMENT
    if count_min == counter[MAX]:  # we cannot have more domains equal to a
        for x_i in x:
            if x_i[MIN] == a and x_i[MAX] > a:
                x_i[MIN] = a + 1
            elif x_i[MIN] < a and x_i[MAX] == a:
                x_i[MAX] = a - 1
    if count_max == counter[MIN]:  # we cannot have more domains different from a
        for x_i in x:
            if x_i[MIN] <= a <= x_i[MAX]:
                x_i[:] = a
    return PROP_CONSISTENCY
