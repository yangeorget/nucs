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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_nvalue(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n


@njit(cache=True, fastmath=True)
def get_triggers_nvalue(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_nvalue(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`y = |\\{x_0, ..., x_{n-1}\\}|`, the number of distinct values taken by the x_i.

    Full domain consistency is NP-hard, so this bounds the count variable y between the maximum number of
    pairwise-disjoint domains (a guaranteed-distinct lower bound) and min(n, |union of the domains|), and
    handles the all-equal case y = 1 exactly.

    :param domains: the domains of the variables, x is the first n-1 domains, y is the last domain
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) - 1
    x = domains[:n]
    y = domains[n]
    if n == 0:  # no variables: zero distinct values
        if y[MIN] > 0 or y[MAX] < 0:
            return PROP_INCONSISTENCY
        y[MIN] = 0
        y[MAX] = 0
        return PROP_CONSISTENCY
    # lower bound: the maximum number of pairwise-disjoint domains must take distinct values
    # (interval-scheduling greedy over domains sorted by their upper bound)
    order_max = np.argsort(x[:, MAX])
    low = 0
    last_end = 0
    for k in range(n):
        i = order_max[k]
        if low == 0 or x[i, MIN] > last_end:
            low += 1
            last_end = x[i, MAX]
    # upper bound: at most min(n, number of integers in the union of the domains) distinct values exist
    order_min = np.argsort(x[:, MIN])
    union = 0
    cur_lo = x[order_min[0], MIN]
    cur_hi = x[order_min[0], MAX]
    for k in range(1, n):
        i = order_min[k]
        if x[i, MIN] > cur_hi + 1:  # gap -> close the current merged block
            union += cur_hi - cur_lo + 1
            cur_lo = x[i, MIN]
            cur_hi = x[i, MAX]
        elif x[i, MAX] > cur_hi:
            cur_hi = x[i, MAX]
    union += cur_hi - cur_lo + 1
    up = n if n < union else union
    if y[MIN] < low:
        y[MIN] = low
    if y[MAX] > up:
        y[MAX] = up
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    if y[MAX] == 1:  # a single distinct value: every x_i must be equal
        lo = x[0, MIN]
        hi = x[0, MAX]
        for i in range(1, n):
            if x[i, MIN] > lo:
                lo = x[i, MIN]
            if x[i, MAX] < hi:
                hi = x[i, MAX]
        if lo > hi:
            return PROP_INCONSISTENCY
        for i in range(n):
            x[i, MIN] = lo
            x[i, MAX] = hi
    return PROP_CONSISTENCY
