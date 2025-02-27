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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_count_leq_c(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


@njit(cache=True)
def get_triggers_count_leq_c(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_count_leq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a) <= c.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is the first parameter, c is the second parameter
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    a = parameters[0]
    c = parameters[1]
    count_max = len(domains)
    count_min = 0
    for domain in domains:
        if domain[MIN] > a or domain[MAX] < a:
            count_max -= 1
        elif domain[MIN] == a and domain[MAX] == a:
            count_min += 1
            if count_min > c:
                return PROP_INCONSISTENCY
    if count_max <= c:
        return PROP_ENTAILMENT
    if count_min == c:  # we cannot have more domains equal to a
        all_different = True
        for domain in domains:
            if domain[MIN] == a:
                if domain[MAX] > a:
                    domain[MIN] = a + 1
            elif domain[MIN] < a:
                if domain[MAX] == a:
                    domain[MAX] = a - 1
                elif domain[MAX] > a:
                    all_different = False
        if all_different:
            return PROP_ENTAILMENT
    return PROP_CONSISTENCY
