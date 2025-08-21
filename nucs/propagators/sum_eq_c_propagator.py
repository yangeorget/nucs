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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_sum_eq_c(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


@njit(cache=True)
def get_triggers_sum_eq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_sum_eq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i x_i = c.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, c is the first parameter
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    n = len(domains)
    domain_sum_min = -parameters[0] + domains[:, MAX].sum()
    domain_sum_max = -parameters[0] + domains[:, MIN].sum()
    for i in range(n):
        new_min = domains[i, MAX] - domain_sum_min
        new_max = domains[i, MIN] - domain_sum_max
        if new_min > domains[i, MIN]:
            domains[i, MIN] = new_min
        if new_max < domains[i, MAX]:
            domains[i, MAX] = new_max
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
