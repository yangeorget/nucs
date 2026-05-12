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
    EVENT_MASK_MAX,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_sum_geq_c(n: int, parameters: NDArray) -> int:
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
def get_triggers_sum_geq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MAX


@njit(cache=True, fastmath=True)
def compute_domains_sum_geq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\sum_i x_i >= c`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, c is the first parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    domain_sum_min = domain_sum_max = -int(parameters[0])
    for i in range(n):
        domain_sum_max += domains[i, MIN]
        domain_sum_min += domains[i, MAX]
    if domain_sum_max >= 0:
        return PROP_ENTAILMENT
    for i in range(n):
        new_min = domains[i, MAX] - domain_sum_min
        if new_min > domains[i, MIN]:
            domains[i, MIN] = new_min
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
