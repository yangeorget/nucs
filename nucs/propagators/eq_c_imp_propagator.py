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


def get_complexity_eq_c_imp(n: int, parameters: NDArray) -> int:
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
def get_triggers_eq_c_imp(n: int, variable: int, parameters: NDArray) -> int:
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
    # implication vacuous, so there is nothing to deduce. x still needs both bounds.
    return EVENT_MASK_MIN if variable == 0 else EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_eq_c_imp(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the half-reified (implied) constraint :math:`b \\rightarrow x = c` for a constant c.

    Unlike the fully-reified :math:`b \\Leftrightarrow x = c`, this only enforces the equality when b is true
    and the contrapositive (b becomes false when x = c is impossible); it never forces b true on entailment nor
    removes c from x when b is false, so it wakes and filters strictly less.

    :param domains: the domains of the variables, b is the first domain, x the second
    :type domains: NDArray
    :param parameters: c is the first parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    c = int(parameters[0])
    if b[MAX] == 0:  # b is false: the implication is vacuously satisfied
        return PROP_ENTAILMENT
    if b[MIN] == 1:  # b is true: enforce x = c
        if c < x[MIN] or c > x[MAX]:
            return PROP_INCONSISTENCY
        x[:] = c
        return PROP_ENTAILMENT
    # b is free: only the contrapositive can fire
    if x[MIN] > c or x[MAX] < c:  # x = c is impossible -> b must be false
        b[:] = 0
        return PROP_ENTAILMENT
    if x[MIN] == c and x[MAX] == c:  # x = c is entailed -> the implication holds for any b
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
