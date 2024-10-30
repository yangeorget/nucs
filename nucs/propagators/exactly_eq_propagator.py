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


def get_complexity_exactly_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 2 * n


def get_triggers_exactly_eq(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.ones((n, 2), dtype=np.bool)


@njit(cache=True)
def compute_domains_exactly_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a) = c.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is the first parameter, c is the second parameter
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    a = parameters[0]
    c = parameters[1]
    count_max = len(domains) - c
    count_min = -c
    for domain in domains:
        if domain[MIN] > a or domain[MAX] < a:
            count_max -= 1
            if count_max < 0:
                return PROP_INCONSISTENCY
        elif domain[MIN] == a and domain[MAX] == a:
            count_min += 1
            if count_min > 0:
                return PROP_INCONSISTENCY
    if count_min == 0 and count_max == 0:
        return PROP_ENTAILMENT
    if count_min == 0:  # we cannot have more domains equal to a
        for domain in domains:
            if domain[MIN] == a and domain[MAX] > a:
                domain[MIN] = a + 1
            if domain[MIN] < a and domain[MAX] == a:
                domain[MAX] = a - 1
    elif count_max == 0:  # we cannot have more domains different from a
        for domain in domains:
            if domain[MIN] <= a <= domain[MAX]:
                domain[:] = a
    return PROP_CONSISTENCY
