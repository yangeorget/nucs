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

from nucs.constants import (
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_max_leq(n: int, parameters: NDArray) -> int:
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
def get_triggers_max_leq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN if variable < n - 1 else EVENT_MASK_MAX


@njit(cache=True, fastmath=True)
def compute_domains_max_leq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\max_i x_i <= x_{n-1}`.

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
    # max_i x_i <= y means every x_i <= y, and y >= max_i x_i. The constraint is entailed once even
    # the largest x_i[MAX] cannot exceed y, so that cheap test is done first in its own short loop.
    y_min = y[MIN]
    y_max = y[MAX]
    max_x_max = x[0, MAX]
    for i in range(1, n):
        if x[i, MAX] > max_x_max:
            max_x_max = x[i, MAX]
    if max_x_max <= y_min:
        return PROP_ENTAILMENT
    # Otherwise a single fused loop caps each x_i[MAX] at y_max (with an inline inconsistency check)
    # while accumulating max_x_min, which then raises y[MIN]; no second pass over x is needed.
    max_x_min = x[0, MIN]
    for i in range(n):
        if x[i, MIN] > max_x_min:
            max_x_min = x[i, MIN]
        if y_max < x[i, MAX]:
            x[i, MAX] = y_max
            if y_max < x[i, MIN]:
                return PROP_INCONSISTENCY
    if max_x_min > y_min:
        y[MIN] = max_x_min
        if max_x_min > y_max:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
