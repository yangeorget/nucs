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

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_element_l_eq_c_alldifferent(n: int, parameters: NDArray) -> int:
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
def get_triggers_element_l_eq_c_alldifferent(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_element_l_eq_c_alldifferent(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces :math:`l_i = c` when the elements of l are all different.

    :param domains: the domains of the variables, l is the list of the first n-1 domains, i is the last domain
    :type domains: NDArray
    :param parameters: the parameters of the propagator, c is the first parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    l = domains[:-1]
    i = domains[-1]
    c = int(parameters[0])
    # i could be updated only once
    i[DOMAIN_MIN] = max(i[DOMAIN_MIN], 0)
    i[DOMAIN_MAX] = min(i[DOMAIN_MAX], len(l) - 1)
    non_intersecting_idx = -1
    for idx in range(i[DOMAIN_MIN], i[DOMAIN_MAX] + 1):
        if c < l[idx, DOMAIN_MIN] or c > l[idx, DOMAIN_MAX]:  # no intersection
            if non_intersecting_idx == -1:
                non_intersecting_idx = idx
            if idx == i[DOMAIN_MIN]:
                i[DOMAIN_MIN] += 1
        else:  # intersection
            if c == l[idx, DOMAIN_MIN] and c == l[idx, DOMAIN_MAX]:
                i[:] = idx
                return PROP_ENTAILMENT
            non_intersecting_idx = -1
    if non_intersecting_idx >= 0:
        i[DOMAIN_MAX] = non_intersecting_idx - 1
        if i[DOMAIN_MAX] < i[DOMAIN_MIN]:
            return PROP_INCONSISTENCY
    if i[DOMAIN_MIN] == i[DOMAIN_MAX]:
        l[i[DOMAIN_MIN]] = c
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
