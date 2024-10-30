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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_element_lic(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


def get_triggers_element_lic(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.ones((n, 2), dtype=np.bool)


@njit(cache=True)
def compute_domains_element_lic(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces l_i = c.
    :param domains: the domains of the variables, l is the list of the first n-1 domains, i is the last domain
    :param parameters: the parameters of the propagator, c is the first parameter
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    l = domains[:-1]
    i = domains[-1]
    # update i
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    if i[MAX] < i[MIN]:
        return PROP_INCONSISTENCY
    c = parameters[0]
    i_min = i[MIN]
    i_max = i[MAX]
    for idx in range(i[MIN], i[MAX] + 1):
        if c < l[idx, MIN] or c > l[idx, MAX]:  # no intersection
            if idx == i_min:
                i_min += 1
            elif idx == i_max:
                i_max -= 1
    if i_max < i_min:
        return PROP_INCONSISTENCY
    i[MIN] = i_min
    i[MAX] = i_max
    # when strict, update l
    # for idx in range(0, i[MIN]):
    #     if l[idx, MIN] == c:
    #         l[idx, MIN] = c + 1
    #     elif l[idx, MAX] == c:
    #         l[idx, MAX] = c - 1
    #     if l[idx, MIN] > l[idx, MAX]:
    #         return PROP_INCONSISTENCY
    # for idx in range(i[MAX] + 1, len(l)):
    #     if l[idx, MIN] == c:
    #         l[idx, MIN] = c + 1
    #     elif l[idx, MAX] == c:
    #         l[idx, MAX] = c - 1
    #     if l[idx, MIN] > l[idx, MAX]:
    #         return PROP_INCONSISTENCY
    if i_min == i_max:
        l[i_min, MIN] = l[i_max, MAX] = c
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
