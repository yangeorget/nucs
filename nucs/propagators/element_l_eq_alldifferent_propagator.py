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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_element_l_eq_alldifferent(n: int, parameters: NDArray) -> int:
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
def get_triggers_element_l_eq_alldifferent(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_element_l_eq_alldifferent(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces :math:`l_i = v` when alldifferent(l).

    :param domains: the domains of the variables,
           l is the list of the first n-2 domains,
           i is the (n-1)th domain,
           v is the last domain
    :type domains: NDArray
    :param parameters: the parameters of the propagator, it is unused
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    l = domains[:-2]
    i = domains[-2]
    v = domains[-1]
    # i could be updated only once
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    l_v_min = sys.maxsize
    l_v_max = -sys.maxsize
    old_v_min = v[MIN]
    old_v_max = v[MAX]
    non_intersecting_idx = -1
    if old_v_min == old_v_max:
        for idx in range(i[MIN], i[MAX] + 1):
            l_idx_min = l[idx, MIN]
            l_idx_max = l[idx, MAX]
            if old_v_max < l_idx_min or old_v_min > l_idx_max:  # no intersection
                if non_intersecting_idx == -1:
                    non_intersecting_idx = idx
                if idx == i[MIN]:
                    i[MIN] += 1
            else:  # intersection
                if l_idx_min == l_idx_max and old_v_min == l_idx_min:
                    i[:] = idx
                    return PROP_ENTAILMENT
                non_intersecting_idx = -1
                if l_idx_min < l_v_min:
                    l_v_min = l_idx_min
                if l_idx_max > l_v_max:
                    l_v_max = l_idx_max
    else:
        for idx in range(i[MIN], i[MAX] + 1):
            l_idx_min = l[idx, MIN]
            l_idx_max = l[idx, MAX]
            if old_v_max < l_idx_min or old_v_min > l_idx_max:  # no intersection
                if non_intersecting_idx == -1:
                    non_intersecting_idx = idx
                if idx == i[MIN]:
                    i[MIN] += 1
            else:  # intersection
                non_intersecting_idx = -1
                if l_idx_min < l_v_min:
                    l_v_min = l_idx_min
                if l_idx_max > l_v_max:
                    l_v_max = l_idx_max
    if non_intersecting_idx >= 0:
        i[MAX] = non_intersecting_idx - 1
        if i[MAX] < i[MIN]:
            return PROP_INCONSISTENCY
    if l_v_min > old_v_min:
        v[MIN] = l_v_min
    if l_v_max < old_v_max:
        v[MAX] = l_v_max
    if i[MIN] == i[MAX]:
        l[i[MIN]] = v
        if v[MIN] == v[MAX]:
            return PROP_ENTAILMENT
    return PROP_CONSISTENCY
