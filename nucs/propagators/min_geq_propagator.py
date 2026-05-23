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


def get_complexity_min_geq(n: int, parameters: NDArray) -> int:
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
def get_triggers_min_geq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MAX if variable < n - 1 else EVENT_MASK_MIN


@njit(cache=True, fastmath=True)
def compute_domains_min_geq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\min_i x_i >= x_{n-1}`.

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
    # min_i x_i >= y means every x_i >= y, and y <= min_i x_i. The constraint is entailed once even
    # the smallest x_i[MIN] cannot fall below y, so that cheap test is done first in its own short loop.
    y_min = y[MIN]
    y_max = y[MAX]
    min_x_min = x[0, MIN]
    for i in range(1, n):
        if x[i, MIN] < min_x_min:
            min_x_min = x[i, MIN]
    if y_max <= min_x_min:
        return PROP_ENTAILMENT
    # Otherwise a single fused loop raises each x_i[MIN] to y_min (with an inline inconsistency check)
    # while accumulating min_x_max, which then lowers y[MAX]; no second pass over x is needed.
    min_x_max = x[0, MAX]
    for i in range(n):
        if x[i, MAX] < min_x_max:
            min_x_max = x[i, MAX]
        if y_min > x[i, MIN]:
            x[i, MIN] = y_min
            if x[i, MAX] < y_min:
                return PROP_INCONSISTENCY
    if min_x_max < y_max:
        y[MAX] = min_x_max
        if y_min > min_x_max:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
