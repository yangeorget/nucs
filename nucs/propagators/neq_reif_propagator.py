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
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


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
    if b[DOMAIN_MIN] == 1:
        # Check if x and y are already different
        if x[DOMAIN_MAX] < y[DOMAIN_MIN] or y[DOMAIN_MAX] < x[DOMAIN_MIN]:
            return PROP_ENTAILMENT
        # If x is fixed, remove that value from y
        if x[DOMAIN_MIN] == x[DOMAIN_MAX]:
            if y[DOMAIN_MIN] == x[DOMAIN_MIN]:
                y[DOMAIN_MIN] += 1
                if y[DOMAIN_MIN] > y[DOMAIN_MAX]:
                    return PROP_INCONSISTENCY
            if y[DOMAIN_MAX] == x[DOMAIN_MAX]:
                y[DOMAIN_MAX] -= 1
                if y[DOMAIN_MIN] > y[DOMAIN_MAX]:
                    return PROP_INCONSISTENCY
        # If y is fixed, remove that value from x
        if y[DOMAIN_MIN] == y[DOMAIN_MAX]:
            if x[DOMAIN_MIN] == y[DOMAIN_MIN]:
                x[DOMAIN_MIN] += 1
                if x[DOMAIN_MIN] > x[DOMAIN_MAX]:
                    return PROP_INCONSISTENCY
            if x[DOMAIN_MAX] == y[DOMAIN_MAX]:
                x[DOMAIN_MAX] -= 1
                if x[DOMAIN_MIN] > x[DOMAIN_MAX]:
                    return PROP_INCONSISTENCY
        return PROP_CONSISTENCY
    # If b is fixed to 0, then x = y
    if b[DOMAIN_MAX] == 0:
        # Compute intersection
        new_min = max(x[DOMAIN_MIN], y[DOMAIN_MIN])
        new_max = min(x[DOMAIN_MAX], y[DOMAIN_MAX])
        if new_min > new_max:
            return PROP_INCONSISTENCY
        x[DOMAIN_MIN] = y[DOMAIN_MIN] = new_min
        x[DOMAIN_MAX] = y[DOMAIN_MAX] = new_max
        if x[DOMAIN_MIN] == x[DOMAIN_MAX]:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # If x and y have no overlap, then b = 1
    if x[DOMAIN_MAX] < y[DOMAIN_MIN] or y[DOMAIN_MAX] < x[DOMAIN_MIN]:
        b[:] = 1
        return PROP_ENTAILMENT
    # If x and y are fixed, then b = 1 or 0
    if x[DOMAIN_MIN] == x[DOMAIN_MAX] and y[DOMAIN_MIN] == y[DOMAIN_MAX]:
        b[:] = 1 if x[DOMAIN_MIN] != y[DOMAIN_MIN] else 0
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
