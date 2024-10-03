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
# Copyright 2024 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_exactly_true(n: int, parameters: NDArray) -> float:
    return 2 * n


def get_triggers_exactly_true(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_exactly_true(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (b_i == 1) = c when for each i, b_i is a boolean variable.
    :param domains: the domains of the variables, b is an alias for domains
    :param parameters: the parameters of the propagator, c is the first parameter
    """
    c = parameters[0]
    count_max = len(domains) - c
    count_min = -c
    for domain in domains:
        if domain[MAX] < 1:
            count_max -= 1
            if count_max < 0:
                return PROP_INCONSISTENCY
        elif domain[MIN] == 1 and domain[MAX] == 1:
            count_min += 1
            if count_min > 0:
                return PROP_INCONSISTENCY
    if count_min == 0 and count_max == 0:
        return PROP_ENTAILMENT
    if count_min == 0:  # we cannot have more domains equal to 1
        for domain in domains:
            if domain[MIN] == 0 and domain[MAX] == 1:
                domain[MAX] = 0
    elif count_max == 0:  # we cannot have more domains different from 1
        for domain in domains:
            if domain[MIN] == 0 and domain[MAX] == 1:
                domain[MIN] = 1
    return PROP_CONSISTENCY
