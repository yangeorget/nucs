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

from nucs.constants import EVENT_MASK_GROUND, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_neq(n: int, parameters: NDArray) -> int:
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


@njit(cache=True, fastmath=True)
def get_triggers_neq(n: int, variable: int, parameters: NDArray) -> int:
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
    return EVENT_MASK_GROUND


@njit(cache=True, fastmath=True)
def compute_domains_neq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x \\neq y`.

    :param domains: the domains of the variables, x is the first domain, y the second
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    # Disjoint domains: x != y can never be violated.
    if x[MAX] < y[MIN] or y[MAX] < x[MIN]:
        return PROP_ENTAILMENT
    # Bound consistency can only filter when a variable is bound: the forbidden value is then
    # removed from the other domain, but solely when it sits on one of its bounds (no holes).
    if x[MIN] == x[MAX]:
        if y[MIN] == x[MIN]:
            y[MIN] += 1
        elif y[MAX] == x[MIN]:
            y[MAX] -= 1
        else:
            return PROP_CONSISTENCY
        if y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT
    if y[MIN] == y[MAX]:
        if x[MIN] == y[MIN]:
            x[MIN] += 1
        elif x[MAX] == y[MIN]:
            x[MAX] -= 1
        else:
            return PROP_CONSISTENCY
        if x[MIN] > x[MAX]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
