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
    EVENT_MASK_MIN_MAX,
    EVENT_MASK_NONE,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_affine_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_affine_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX if parameters[variable] != 0 else EVENT_MASK_NONE


@njit(cache=True, fastmath=True)
def compute_domains_affine_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\sum_i a_i * x_i = a_{n}`.

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
        unbound_count = 0
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
            if factor != 0 and x_min < x_max:
                unbound_count += 1
        if unbound_count == 0:
            return PROP_ENTAILMENT if domain_sum_min == 0 else PROP_INCONSISTENCY
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
                new_max = x_min + (-domain_sum_max // factor)
            else:
                new_min = x_max - (domain_sum_max // factor)
                new_max = x_min + (-domain_sum_min // factor)
            if new_min > x_min:
                domains[i, MIN] = new_min
                has_changed = True
            if new_max < x_max:
                domains[i, MAX] = new_max
                has_changed = True
            if domains[i, MIN] > domains[i, MAX]:
                return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
