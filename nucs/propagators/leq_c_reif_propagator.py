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
    if b[DOMAIN_MIN] == 1:
        x[DOMAIN_MAX] = min(x[DOMAIN_MAX], y[DOMAIN_MAX] + c)
        y[DOMAIN_MIN] = max(y[DOMAIN_MIN], x[DOMAIN_MIN] - c)
        if x[DOMAIN_MIN] > x[DOMAIN_MAX] or y[DOMAIN_MIN] > y[DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        # The relation can no longer be violated.
        if x[DOMAIN_MAX] <= y[DOMAIN_MIN] + c:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # b is fixed to 0: enforce the negation x >= y + c + 1.
    if b[DOMAIN_MAX] == 0:
        x[DOMAIN_MIN] = max(x[DOMAIN_MIN], y[DOMAIN_MIN] + c + 1)
        y[DOMAIN_MAX] = min(y[DOMAIN_MAX], x[DOMAIN_MAX] - c - 1)
        if x[DOMAIN_MIN] > x[DOMAIN_MAX] or y[DOMAIN_MIN] > y[DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        # The negated relation can no longer be violated.
        if x[DOMAIN_MIN] >= y[DOMAIN_MAX] + c + 1:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # b is free: fix it as soon as the relation is decided by the bounds of x and y.
    if x[DOMAIN_MAX] <= y[DOMAIN_MIN] + c:
        b[:] = 1
        return PROP_ENTAILMENT
    if x[DOMAIN_MIN] > y[DOMAIN_MAX] + c:
        b[:] = 0
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
