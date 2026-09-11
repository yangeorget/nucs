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
from collections.abc import Sequence

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


def get_state_element_l_eq_alldifferent(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


@njit(cache=True)
def get_triggers_element_l_eq_alldifferent(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_element_l_eq_alldifferent(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Enforces :math:`l_i = v` when alldifferent(l).

    :param domains: the domains of the variables,
           l is the list of the first n-2 domains,
           i is the (n-1)th domain,
           v is the last domain
    :type domains: NDArray
    :param parameters: the parameters of the propagator, it is unused
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    l = domains[:-2]
    i = domains[-2]
    v = domains[-1]
    # i and v are the only two domains this can narrow, and old_v_min/old_v_max below already snapshot v,
    # so snapshotting i too is what the change report costs -- against a dozen write sites, several of
    # them inside the two scanning loops. The one write to l is tested where it happens.
    old_i_min = i[DOMAIN_MIN]
    old_i_max = i[DOMAIN_MAX]
    l_changed = False
    # i could be updated only once
    i[DOMAIN_MIN] = max(i[DOMAIN_MIN], 0)
    i[DOMAIN_MAX] = min(i[DOMAIN_MAX], len(l) - 1)
    l_v_min = sys.maxsize
    l_v_max = -sys.maxsize
    old_v_min = v[DOMAIN_MIN]
    old_v_max = v[DOMAIN_MAX]
    non_intersecting_idx = -1
    if old_v_min == old_v_max:
        for idx in range(i[DOMAIN_MIN], i[DOMAIN_MAX] + 1):
            l_idx_min = l[idx, DOMAIN_MIN]
            l_idx_max = l[idx, DOMAIN_MAX]
            if old_v_max < l_idx_min or old_v_min > l_idx_max:  # no intersection
                if non_intersecting_idx == -1:
                    non_intersecting_idx = idx
                if idx == i[DOMAIN_MIN]:
                    i[DOMAIN_MIN] += 1
            else:  # intersection
                if l_idx_min == l_idx_max and old_v_min == l_idx_min:
                    i[:] = idx
                    return PROP_ENTAILMENT
                non_intersecting_idx = -1
                l_v_min = min(l_v_min, l_idx_min)
                l_v_max = max(l_v_max, l_idx_max)
    else:
        for idx in range(i[DOMAIN_MIN], i[DOMAIN_MAX] + 1):
            l_idx_min = l[idx, DOMAIN_MIN]
            l_idx_max = l[idx, DOMAIN_MAX]
            if old_v_max < l_idx_min or old_v_min > l_idx_max:  # no intersection
                if non_intersecting_idx == -1:
                    non_intersecting_idx = idx
                if idx == i[DOMAIN_MIN]:
                    i[DOMAIN_MIN] += 1
            else:  # intersection
                non_intersecting_idx = -1
                l_v_min = min(l_v_min, l_idx_min)
                l_v_max = max(l_v_max, l_idx_max)
    if non_intersecting_idx >= 0:
        i[DOMAIN_MAX] = non_intersecting_idx - 1
        if i[DOMAIN_MAX] < i[DOMAIN_MIN]:
            return PROP_INCONSISTENCY
    if l_v_min > old_v_min:
        v[DOMAIN_MIN] = l_v_min
    if l_v_max < old_v_max:
        v[DOMAIN_MAX] = l_v_max
    if i[DOMAIN_MIN] == i[DOMAIN_MAX]:
        idx = i[DOMAIN_MIN]
        if l[idx, DOMAIN_MIN] != v[DOMAIN_MIN] or l[idx, DOMAIN_MAX] != v[DOMAIN_MAX]:
            l[idx] = v
            l_changed = True
        if v[DOMAIN_MIN] == v[DOMAIN_MAX]:
            return PROP_ENTAILMENT
    if not (
        l_changed
        or i[DOMAIN_MIN] != old_i_min
        or i[DOMAIN_MAX] != old_i_max
        or v[DOMAIN_MIN] != old_v_min
        or v[DOMAIN_MAX] != old_v_max
    ):
        prop_state[0] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
