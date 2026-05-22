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
    EVENT_MASK_MIN,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    EVENT_MASK_NONE,
)


def get_complexity_affine_geq(n: int, parameters: NDArray) -> int:
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
def get_triggers_affine_geq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    if parameters[variable] == 0:
        return EVENT_MASK_NONE
    return EVENT_MASK_MIN if parameters[variable] < 0 else EVENT_MASK_MAX


@njit(cache=True, fastmath=True)
def compute_domains_affine_geq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\sum_i a_i * x_i >= a_{n}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is an alias for parameters
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    factors = parameters[:-1]
    n = len(factors)
    has_changed = True
    while has_changed:
        has_changed = False
        domain_sum_min = domain_sum_max = -parameters[-1]
        for i in range(n):
            factor = factors[i]
            x_min = domains[i, MIN]
            x_max = domains[i, MAX]
            if factor > 0:
                domain_sum_min += factor * x_max
                domain_sum_max += factor * x_min
            else:
                domain_sum_min += factor * x_min
                domain_sum_max += factor * x_max
        if domain_sum_max >= 0:
            return PROP_ENTAILMENT
        if domain_sum_min < 0:
            return PROP_INCONSISTENCY
        for i in range(n):
            factor = factors[i]
            if factor == 0:
                continue
            x_min = domains[i, MIN]
            x_max = domains[i, MAX]
            if x_min == x_max:
                continue
            if factor > 0:
                new_min = x_max - (domain_sum_min // factor)
                if new_min > x_min:
                    domains[i, MIN] = new_min
                    has_changed = True
            else:
                new_max = x_min + (-domain_sum_min // factor)
                if new_max < x_max:
                    domains[i, MAX] = new_max
                    has_changed = True
    return PROP_CONSISTENCY
