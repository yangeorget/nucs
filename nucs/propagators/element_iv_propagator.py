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
import sys
from typing import List

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_element_iv(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


def get_triggers_element_iv(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.full(n, dtype=np.uint8, fill_value=EVENT_MASK_MIN_MAX)


@njit(cache=True)
def compute_domains_element_iv(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces l_i = v.
    :param domains: the domains of the variables,
           i is the first domain,
           v is the last domain
    :param parameters: the parameters of the propagator
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    l = parameters
    i = domains[0]
    v = domains[1]
    # i could be updated only once
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    v_min = sys.maxsize
    v_max = -sys.maxsize
    indices: List[int] = []
    for idx in range(i[MIN], i[MAX] + 1):
        if v[MAX] < l[idx] or v[MIN] > l[idx]:  # no intersection
            indices.append(idx)
            if idx == i[MIN]:
                i[MIN] += 1
        else:  # intersection
            if l[idx] < v_min:
                v_min = l[idx]
            if l[idx] > v_max:
                v_max = l[idx]
    for ix in range(len(indices) - 1, -1, -1):
        if indices[ix] != i[MAX]:
            break
        i[MAX] -= 1
    if i[MAX] < i[MIN]:
        return PROP_INCONSISTENCY
    v[MIN] = max(v[MIN], v_min)
    v[MAX] = min(v[MAX], v_max)
    if i[MIN] == i[MAX]:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
