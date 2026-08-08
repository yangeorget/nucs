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


def get_complexity_leq_c_reif(n: int, parameters: NDArray) -> int:
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
def get_triggers_leq_c_reif(n: int, variable: int, parameters: NDArray) -> int:
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
def advise_leq_c_reif(domains: NDArray, parameters: NDArray) -> bool:
    """
    Advisor for :math:`b \\Leftrightarrow x \\leq y + a_0`: when b is true the inequality can tighten x from y
    (or y from x); when b is false the negation x > y + a_0 can tighten; when b is free b can be decided only
    when the relation is already entailed (x <= y + a_0) or disentailed (x > y + a_0).

    :param domains: the domains of the variables, b is the first domain, x the second, y the third
    :type domains: NDArray
    :param parameters: the constant a_0 added to y, parameters[0]
    :type parameters: NDArray

    :return: whether the propagator should be scheduled
    :rtype: bool
    """
    b = domains[0]
    x = domains[1]
    y = domains[2]
    c = parameters[0]
    if b[MIN] == 1:  # b true: x <= y + c
        return x[MAX] > y[MAX] + c or x[MIN] - c > y[MIN]
    if b[MAX] == 0:  # b false: x > y + c
        return x[MIN] < y[MIN] + c + 1 or y[MAX] > x[MAX] - c - 1
    return x[MAX] <= y[MIN] + c or x[MIN] > y[MAX] + c  # b free: relation decided


@njit(cache=True)
def compute_domains_leq_c_reif(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`b <=> x \\leq y + a_0`.

    :param domains: the domains of the variables, b is the first domain, x is the second, y is the third
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a_0 is the constant added to y
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    y = domains[2]
    c = parameters[0]
    # b is fixed to 1: enforce x <= y + c.
    if b[MIN] == 1:
        if y[MAX] + c < x[MAX]:
            x[MAX] = y[MAX] + c
        if x[MIN] - c > y[MIN]:
            y[MIN] = x[MIN] - c
        if x[MIN] > x[MAX] or y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
        # The relation can no longer be violated.
        if x[MAX] <= y[MIN] + c:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # b is fixed to 0: enforce the negation x >= y + c + 1.
    if b[MAX] == 0:
        if y[MIN] + c + 1 > x[MIN]:
            x[MIN] = y[MIN] + c + 1
        if x[MAX] - c - 1 < y[MAX]:
            y[MAX] = x[MAX] - c - 1
        if x[MIN] > x[MAX] or y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
        # The negated relation can no longer be violated.
        if x[MIN] >= y[MAX] + c + 1:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # b is free: fix it as soon as the relation is decided by the bounds of x and y.
    if x[MAX] <= y[MIN] + c:
        b[:] = 1
        return PROP_ENTAILMENT
    if x[MIN] > y[MAX] + c:
        b[:] = 0
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
