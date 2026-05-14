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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_count_leq_c(n: int, parameters: NDArray) -> int:
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
def get_triggers_count_leq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_count_leq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`S\\sum_i (x_i == a) <= c`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is the first parameter, c is the second parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    a = int(parameters[0])
    c = int(parameters[1])
    count_max = len(domains)
    count_min = 0
    for domain in domains:
        domain_min = domain[MIN]
        domain_max = domain[MAX]
        if domain_min > a or domain_max < a:
            count_max -= 1
            if count_max <= c:
                return PROP_ENTAILMENT
        elif domain_min == a and domain_max == a:
            count_min += 1
            if count_min > c:
                return PROP_INCONSISTENCY
    if count_min == c:  # we cannot have more domains equal to a
        all_different = True
        for domain in domains:
            domain_min = domain[MIN]
            domain_max = domain[MAX]
            if domain_min == a:
                if domain_max > a:
                    domain[MIN] = a + 1
            elif domain_min < a:
                if domain_max == a:
                    domain[MAX] = a - 1
                elif domain_max > a:
                    all_different = False
        if all_different:
            return PROP_ENTAILMENT
    return PROP_CONSISTENCY
