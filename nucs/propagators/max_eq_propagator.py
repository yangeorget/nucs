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
# Copyright 2024-2026 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_max_eq(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n


@njit(cache=True, fastmath=True)
def get_triggers_max_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_max_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\max_i x_i = x_{n-1}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[:-1]
    y = domains[-1]
    n = len(x)
    max_x_min = x[0, MIN]
    max_x_max = x[0, MAX]
    for i in range(1, n):
        if x[i, MIN] > max_x_min:
            max_x_min = x[i, MIN]
        if x[i, MAX] > max_x_max:
            max_x_max = x[i, MAX]
    y_min = y[MIN]
    y_max = y[MAX]
    if max_x_min > y_min:
        y_min = max_x_min
        y[MIN] = y_min
    if max_x_max < y_max:
        y_max = max_x_max
        y[MAX] = y_max
    if y_min > y_max:
        return PROP_INCONSISTENCY
    candidates_nb = 0
    candidate_idx = -1
    for i in range(n):
        if x[i, MAX] >= y_max:
            x[i, MAX] = y_max
            candidate_idx = i
            candidates_nb += 1
    if candidates_nb == 1:
        x[candidate_idx, MIN] = y_min
    return PROP_CONSISTENCY
