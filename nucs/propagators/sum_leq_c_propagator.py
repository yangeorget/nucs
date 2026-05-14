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
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_sum_leq_c(n: int, parameters: NDArray) -> int:
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
def get_triggers_sum_leq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN


@njit(cache=True, fastmath=True)
def compute_domains_sum_leq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\sum_i x_i <= c`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, c is the first parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    domain_sum_min = domain_sum_max = -int(parameters[0])
    unbound_count = 0
    for i in range(n):
        x_min = domains[i, MIN]
        x_max = domains[i, MAX]
        domain_sum_min += x_max
        domain_sum_max += x_min
        if x_min < x_max:
            unbound_count += 1
    if domain_sum_min <= 0:
        return PROP_ENTAILMENT
    if unbound_count == 0:
        return PROP_INCONSISTENCY
    for i in range(n):
        x_min = domains[i, MIN]
        x_max = domains[i, MAX]
        if x_min == x_max:
            continue
        new_max = x_min - domain_sum_max
        if new_max < x_max:
            domains[i, MAX] = new_max
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
    return PROP_ENTAILMENT if unbound_count == 1 else PROP_CONSISTENCY
