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


def get_complexity_min_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_min_eq(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_min_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\min_i x_i = x_{n-1}`.

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
    # y = min_i x_i, so y is bounded by [min of the x mins, min of the x maxes]. A single manual
    # loop computes both reductions at once (replacing two np.min passes) and y's bounds are kept in
    # locals to avoid repeated array indexing.
    min_x_min = x[0, MIN]
    min_x_max = x[0, MAX]
    for i in range(1, n):
        if x[i, MIN] < min_x_min:
            min_x_min = x[i, MIN]
        if x[i, MAX] < min_x_max:
            min_x_max = x[i, MAX]
    y_min = y[MIN]
    y_max = y[MAX]
    if min_x_min > y_min:
        y_min = min_x_min
        y[MIN] = y_min
    if min_x_max < y_max:
        y_max = min_x_max
        y[MAX] = y_max
    if y_min > y_max:
        return PROP_INCONSISTENCY
    # No x_i may fall below y, so raise every x_i[MIN] to y_min. If a single x_i can still reach
    # y_min it is the only one able to attain the minimum, hence it must be at most y_max.
    candidates_nb = 0
    candidate_idx = -1
    for i in range(n):
        if x[i, MIN] <= y_min:
            x[i, MIN] = y_min
            candidate_idx = i
            candidates_nb += 1
    if candidates_nb == 1:
        x[candidate_idx, MAX] = y_max
    return PROP_CONSISTENCY
