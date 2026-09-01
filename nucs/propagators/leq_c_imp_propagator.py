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
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_leq_c_imp(n: int, parameters: NDArray) -> int:
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
def get_triggers_leq_c_imp(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_leq_c_imp(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the half-reified (implied) constraint :math:`b \\rightarrow x \\leq y + a_0`.

    Unlike the fully-reified :math:`b \\Leftrightarrow x \\leq y + a_0`, this only enforces the inequality when b
    is true and the contrapositive (b becomes false when x <= y + a_0 is impossible); it never enforces the
    negation when b is false nor forces b true on entailment, so it wakes and filters strictly less.

    :param domains: the domains of the variables, b is the first domain, x the second, y the third
    :type domains: NDArray
    :param parameters: the constant a_0 added to y, parameters[0]
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    y = domains[2]
    c = parameters[0]
    if b[DOMAIN_MAX] == 0:  # b is false: the implication is vacuously satisfied
        return PROP_ENTAILMENT
    if b[DOMAIN_MIN] == 1:  # b is true: enforce x <= y + c
        x[DOMAIN_MAX] = min(x[DOMAIN_MAX], y[DOMAIN_MAX] + c)
        y[DOMAIN_MIN] = max(y[DOMAIN_MIN], x[DOMAIN_MIN] - c)
        if x[DOMAIN_MIN] > x[DOMAIN_MAX] or y[DOMAIN_MIN] > y[DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        if x[DOMAIN_MAX] <= y[DOMAIN_MIN] + c:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    # b is free: only the contrapositive can fire
    if x[DOMAIN_MIN] > y[DOMAIN_MAX] + c:  # x <= y + c is impossible -> b must be false
        b[:] = 0
        return PROP_ENTAILMENT
    if x[DOMAIN_MAX] <= y[DOMAIN_MIN] + c:  # x <= y + c is entailed -> the implication holds for any b
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
