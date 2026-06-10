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


def get_complexity_member(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the allowed values
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return len(parameters)


@njit(cache=True, fastmath=True)
def get_triggers_member(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param variable: the index of the variable
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_member(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x \\in \\{a_0, ..., a_{n-1}\\}`.

    :param domains: the domains of the variables, x is the first (and only) domain
    :type domains: NDArray
    :param parameters: the allowed values, in strictly ascending order
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    n = len(parameters)
    # Domains are intervals, so filtering is bound-consistent: raise the lower bound to the smallest
    # allowed value that is still in range, lower the upper bound to the largest one. Holes between
    # allowed values cannot be represented and are therefore left in place.
    lo = 0
    while lo < n and parameters[lo] < x[MIN]:
        lo += 1
    hi = n - 1
    while hi >= 0 and parameters[hi] > x[MAX]:
        hi -= 1
    if lo > hi:
        return PROP_INCONSISTENCY
    new_min = parameters[lo]
    new_max = parameters[hi]
    if new_min > x[MIN]:
        x[MIN] = new_min
    if new_max < x[MAX]:
        x[MAX] = new_max
    # The allowed values in range are consecutive integers covering the whole interval: the constraint
    # can never be violated again.
    if hi - lo == new_max - new_min:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
