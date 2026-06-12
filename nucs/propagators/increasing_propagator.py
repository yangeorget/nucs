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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_increasing(n: int, parameters: NDArray) -> int:
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
def get_triggers_increasing(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_increasing(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x_i <= x_{i+1}` for all i.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    # forward sweep: each x_i is at least its predecessor's minimum
    for i in range(1, n):
        if domains[i - 1][MIN] > domains[i][MIN]:
            domains[i][MIN] = domains[i - 1][MIN]
    # backward sweep: each x_i is at most its successor's maximum
    for i in range(n - 2, -1, -1):
        if domains[i + 1][MAX] < domains[i][MAX]:
            domains[i][MAX] = domains[i + 1][MAX]
    # consistency check and entailment detection (entailed when every pair is already ordered)
    entailed = True
    for i in range(n):
        if domains[i][MIN] > domains[i][MAX]:
            return PROP_INCONSISTENCY
        if i < n - 1 and domains[i][MAX] > domains[i + 1][MIN]:
            entailed = False
    return PROP_ENTAILMENT if entailed else PROP_CONSISTENCY
