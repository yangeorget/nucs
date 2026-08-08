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
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_MAX,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_eq_imp(n: int, parameters: NDArray) -> int:
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
def get_triggers_eq_imp(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    # b (variable 0) only needs to wake when it becomes true (b_min rises to 1): b becoming false makes the
    # implication vacuous, so there is nothing to deduce. x and y still need both bounds.
    return EVENT_MASK_MIN if variable == 0 else EVENT_MASK_MIN_MAX


@njit(cache=True)
def advise_eq_imp(domains: NDArray, parameters: NDArray) -> bool:
    """
    Advisor for :math:`b \\rightarrow x = y`: when b is false the implication is vacuous; when b is true x = y
    can tighten unless x and y already share their bounds; when b is free b can be set to false only when x and
    y are disjoint.

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
    if b[MAX] == 0:  # b false: vacuous
        return False
    if b[MIN] == 1:  # b true: x = y
        return x[MIN] != y[MIN] or x[MAX] != y[MAX]
    return x[MAX] < y[MIN] or y[MAX] < x[MIN]  # b free: x = y disentailed -> b = false


@njit(cache=True)
def compute_domains_eq_imp(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the half-reified (implied) constraint :math:`b \\rightarrow x = y`.

    Unlike the fully-reified :math:`b \\Leftrightarrow x = y`, this only enforces the equality when b is true
    and the contrapositive (b becomes false when x and y are disjoint); it never forces b true on entailment nor
    separates x and y when b is false, so it wakes and filters strictly less.

    :param domains: the domains of the variables, b is the first domain, x the second, y the third
    :type domains: NDArray
    :param parameters: unused
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    y = domains[2]
    if b[MAX] == 0:  # b is false: the implication is vacuously satisfied
        return PROP_ENTAILMENT
    if b[MIN] == 1:  # b is true: enforce x = y
        new_min = max(x[MIN], y[MIN])
        new_max = min(x[MAX], y[MAX])
        if new_min > new_max:
            return PROP_INCONSISTENCY
        x[MIN] = y[MIN] = new_min
        x[MAX] = y[MAX] = new_max
        return PROP_ENTAILMENT if new_min == new_max else PROP_CONSISTENCY
    # b is free: only the contrapositive can fire
    if x[MAX] < y[MIN] or y[MAX] < x[MIN]:  # x = y is impossible -> b must be false
        b[:] = 0
        return PROP_ENTAILMENT
    if x[MIN] == x[MAX] and y[MIN] == y[MAX] and x[MIN] == y[MIN]:  # x = y entailed -> holds for any b
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
