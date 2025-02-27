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
# Copyright 2024-2025 - Yan Georget
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
)


def get_complexity_affine_leq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


@njit(cache=True)
def get_triggers_affine_leq(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.
    :param parameters: the parameters
    :return: an array of triggers
    """
    return EVENT_MASK_MAX if parameters[dom_idx] < 0 else EVENT_MASK_MIN


@njit(cache=True)
def compute_domains_affine_leq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i <= a_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is an alias for parameters
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    domain_sum_min = domain_sum_max = -parameters[-1]
    factors = parameters[:-1]
    n = len(factors)
    for i in range(n):
        factor = factors[i]
        if factor > 0:
            domain_sum_min += factor * domains[i, MAX]
            domain_sum_max += factor * domains[i, MIN]
        elif factor < 0:
            domain_sum_min += factor * domains[i, MIN]
            domain_sum_max += factor * domains[i, MAX]
    if domain_sum_min <= 0:
        return PROP_ENTAILMENT
    for i in range(n):
        factor = factors[i]
        if factor != 0:
            if factor > 0:
                domains[i, MAX] = min(domains[i, MAX], domains[i, MIN] + (-domain_sum_max // factor))
            else:
                domains[i, MIN] = max(domains[i, MIN], domains[i, MAX] - (domain_sum_max // factor))
            if domains[i, MIN] > domains[i, MAX]:
                return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
