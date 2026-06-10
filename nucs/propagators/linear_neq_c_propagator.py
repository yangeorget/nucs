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
    EVENT_MASK_GROUND,
    EVENT_MASK_NONE,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_linear_neq_c(n: int, parameters: NDArray) -> int:
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
def get_triggers_linear_neq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator only filters once a single variable remains unbound, so it is woken on ground events.

    :param n: the number of variables
    :type n: int
    :param variable: the index of the variable
    :type variable: int
    :param parameters: the parameters, a is an alias for parameters
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_GROUND if parameters[variable] != 0 else EVENT_MASK_NONE


@njit(cache=True, fastmath=True)
def compute_domains_linear_neq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\sum_i a_i * x_i \\neq a_{n}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is an alias for parameters
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    factors = parameters[:-1]
    c = parameters[-1]
    n = len(factors)
    # sum_min / sum_max bracket the value of the linear expression. unbound is the single unbound
    # variable with a non-zero factor (only meaningful when unbound_count == 1).
    sum_min = sum_max = 0
    unbound_count = 0
    unbound = -1
    for i in range(n):
        factor = factors[i]
        if factor == 0:
            continue
        x_min = domains[i, MIN]
        x_max = domains[i, MAX]
        if factor > 0:
            sum_min += factor * x_min
            sum_max += factor * x_max
        else:
            sum_min += factor * x_max
            sum_max += factor * x_min
        if x_min < x_max:
            unbound_count += 1
            unbound = i
    # If c is outside the reachable range the disequality can never be violated.
    if c < sum_min or c > sum_max:
        return PROP_ENTAILMENT
    if unbound_count == 0:
        # The expression is a fixed value sitting inside [sum_min, sum_max] == {c}.
        return PROP_INCONSISTENCY
    if unbound_count > 1:
        return PROP_CONSISTENCY
    # Exactly one unbound variable: the rest contribute a fixed amount, so the forbidden value of the
    # unbound variable is v = (c - rest) / factor, removable only when it lands on one of its bounds.
    factor = factors[unbound]
    rest = c
    for i in range(n):
        if i != unbound and factors[i] != 0:
            rest -= factors[i] * domains[i, MIN]
    if rest % factor != 0:
        return PROP_ENTAILMENT
    v = rest // factor
    if domains[unbound, MIN] == v:
        domains[unbound, MIN] = v + 1
    elif domains[unbound, MAX] == v:
        domains[unbound, MAX] = v - 1
    else:
        return PROP_CONSISTENCY
    return PROP_ENTAILMENT
