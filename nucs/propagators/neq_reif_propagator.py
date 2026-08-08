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


def get_complexity_neq_reif(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return 1


@njit(cache=True)
def get_triggers_neq_reif(n: int, variable: int, parameters: NDArray) -> int:
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


@njit(cache=True)
def advise_neq_reif(domains: NDArray, parameters: NDArray) -> bool:
    """
    Advisor for :math:`b \\Leftrightarrow x \\neq y`: when b is true (x != y) it can act only when x or y is
    ground or they are disjoint; when b is false (x = y) the intersection can tighten unless x and y already
    share their bounds; when b is free b can be decided only when x and y are disjoint or both ground.

    :param domains: the domains of the variables, b is the first domain, x the second, y the third
    :type domains: NDArray
    :param parameters: unused
    :type parameters: NDArray

    :return: whether the propagator should be scheduled
    :rtype: bool
    """
    b = domains[0]
    x = domains[1]
    y = domains[2]
    if b[MIN] == 1:  # b true: x != y
        return x[MIN] == x[MAX] or y[MIN] == y[MAX] or x[MAX] < y[MIN] or y[MAX] < x[MIN]
    if b[MAX] == 0:  # b false: x = y
        return x[MIN] != y[MIN] or x[MAX] != y[MAX]
    return (x[MAX] < y[MIN] or y[MAX] < x[MIN]) or (x[MIN] == x[MAX] and y[MIN] == y[MAX])  # b free


@njit(cache=True)
def compute_domains_neq_reif(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`b <=> x \\neq y`.

    :param domains: the domains of the variables, b is the first domain, x is the second, y is the third
    :type domains: NDArray
    :param parameters: unused
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    y = domains[2]
    # If b is fixed to 1, then x != y
    if b[MIN] == 1:
        # Check if x and y are already different
        if x[MAX] < y[MIN] or y[MAX] < x[MIN]:
            return PROP_ENTAILMENT
        # If x is fixed, remove that value from y
        if x[MIN] == x[MAX]:
            if y[MIN] == x[MIN]:
                y[MIN] += 1
                if y[MIN] > y[MAX]:
                    return PROP_INCONSISTENCY
            if y[MAX] == x[MAX]:
                y[MAX] -= 1
                if y[MIN] > y[MAX]:
                    return PROP_INCONSISTENCY
        # If y is fixed, remove that value from x
        if y[MIN] == y[MAX]:
            if x[MIN] == y[MIN]:
                x[MIN] += 1
                if x[MIN] > x[MAX]:
                    return PROP_INCONSISTENCY
            if x[MAX] == y[MAX]:
                x[MAX] -= 1
                if x[MIN] > x[MAX]:
                    return PROP_INCONSISTENCY
        return PROP_CONSISTENCY
    # If b is fixed to 0, then x = y
    if b[MAX] == 0:
        # Compute intersection
        new_min = max(x[MIN], y[MIN])
        new_max = min(x[MAX], y[MAX])
        if new_min > new_max:
            return PROP_INCONSISTENCY
        x[MIN] = y[MIN] = new_min
        x[MAX] = y[MAX] = new_max
        if x[MIN] == x[MAX]:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # If x and y have no overlap, then b = 1
    if x[MAX] < y[MIN] or y[MAX] < x[MIN]:
        b[:] = 1
        return PROP_ENTAILMENT
    # If x and y are fixed, then b = 1 or 0
    if x[MIN] == x[MAX] and y[MIN] == y[MAX]:
        b[:] = 1 if x[MIN] != y[MIN] else 0
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
