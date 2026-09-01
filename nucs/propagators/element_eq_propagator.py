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
import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_element_eq(n: int, parameters: NDArray) -> int:
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


@njit(cache=True)
def get_triggers_element_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_element_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces :math:`l_i = v`.

    :param domains: the domains of the variables,
           i is the first domain,
           v is the last domain
    :type domains: NDArray
    :param parameters: the parameters of the propagator, l is an alias for parameters
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    l = parameters
    i = domains[0]
    v = domains[1]
    # i could be updated only once
    i[DOMAIN_MIN] = max(i[DOMAIN_MIN], 0)
    i[DOMAIN_MAX] = min(i[DOMAIN_MAX], len(l) - 1)
    l_v_min = sys.maxsize
    l_v_max = -sys.maxsize
    old_v_min = v[DOMAIN_MIN]
    old_v_max = v[DOMAIN_MAX]
    non_intersecting_idx = -1
    for idx in range(i[DOMAIN_MIN], i[DOMAIN_MAX] + 1):
        l_idx = l[idx]
        if old_v_max < l_idx or old_v_min > l_idx:  # no intersection
            if non_intersecting_idx == -1:
                non_intersecting_idx = idx
            if idx == i[DOMAIN_MIN]:
                i[DOMAIN_MIN] += 1
        else:  # intersection
            non_intersecting_idx = -1
            l_v_min = min(l_v_min, l_idx)
            l_v_max = max(l_v_max, l_idx)
    if non_intersecting_idx >= 0:
        i[DOMAIN_MAX] = non_intersecting_idx - 1
        if i[DOMAIN_MAX] < i[DOMAIN_MIN]:
            return PROP_INCONSISTENCY
    if l_v_min > old_v_min:
        v[DOMAIN_MIN] = l_v_min
    if l_v_max < old_v_max:
        v[DOMAIN_MAX] = l_v_max
    if i[DOMAIN_MIN] == i[DOMAIN_MAX]:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
