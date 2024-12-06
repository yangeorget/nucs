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
from typing import List

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


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
    return np.full(n, dtype=np.uint8, fill_value=EVENT_MASK_MIN_MAX)


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
    c = parameters[0]
    # i could be updated only once
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    indices: List[int] = []
    for idx in range(i[MIN], i[MAX] + 1):
        if c < l[idx, MIN] or c > l[idx, MAX]:  # no intersection
            indices.insert(0, idx)
            if idx == i[MIN]:
                i[MIN] += 1
    for idx in indices:
        if idx != i[MAX]:
            break
        i[MAX] -= 1
    if i[MAX] < i[MIN]:
        return PROP_INCONSISTENCY
    if i[MIN] == i[MAX]:
        l[i[MIN]] = c
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
