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
import math

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY

SORT_MAX_N = 64  # above this arity, np.argsort amortizes its fixed cost and beats the insertion sort


def get_complexity_alldifferent(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return int(n * math.log(n))


@njit(cache=True)
def get_triggers_alldifferent(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def path_set(t: NDArray, start: int, end: int, value: int) -> None:
    """
    Sets t[start], t[t[start]], ..., a[idx] to value until a[idx] = end.

    :param t: an array of pointers
    :type t: NDArray
    :param start: an index
    :type start: int
    :param end: an index
    :type end: int
    :param value: a value
    :type value: int
    """
    while (p := start) != end:
        start = t[p]
        t[p] = value


@njit(cache=True)
def path_min(t: NDArray, i: int) -> int:
    """
    Follows i, t[i], t[t[i], ... until it stops decreasing.

    :param t: an array of pointers
    :type t: NDArray
    :param i: an index
    :type i: int

    :return: the index found
    :rtype: int
    """
    while t[i] < i:
        i = t[i]
    return i


@njit(cache=True)
def path_max(t: NDArray, i: int) -> int:
    """
    Follows i, t[i], t[t[i], ... until it stops increasing.

    :param t: an array of pointers
    :type t: NDArray
    :param i: an index
    :type i: int

    :return: the index found
    :rtype: int
    """
    while t[i] > i:
        i = t[i]
    return i


@njit(cache=True)
def update_bounds(
    bounds: NDArray,
    n: int,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    max_sorted_vars: NDArray,
) -> int:
    min_value = domains[min_sorted_vars[0], MIN]
    max_value = domains[max_sorted_vars[0], MAX] + 1
    last = min_value - 2
    bounds[0] = last
    i = j = nb = 0
    while True:
        if i < n and min_value <= max_value:
            if min_value != last:
                nb += 1
                bounds[nb] = last = min_value
            ranks[min_sorted_vars[i], MIN] = nb
            i += 1
            if i < n:
                min_value = domains[min_sorted_vars[i], MIN]
        else:
            if max_value != last:
                nb += 1
                bounds[nb] = last = max_value
            ranks[max_sorted_vars[j], MAX] = nb
            j += 1
            if j == n:
                break
            max_value = domains[max_sorted_vars[j], MAX] + 1
    bounds[nb + 1] = bounds[nb] + 2
    return nb


@njit(cache=True)
def filter_lower(
    n: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    max_sorted_vars: NDArray,
) -> bool:
    for i in range(1, nb + 2):
        i1 = i - 1
        t[i] = h[i] = i1
        d[i] = bounds[i] - bounds[i1]
    for i in range(n):
        x = ranks[max_sorted_vars[i], MIN]
        y = ranks[max_sorted_vars[i], MAX]
        z = path_max(t, x + 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z + 1
            z = path_max(t, t[z])
            t[z] = j
        delta = d[z] + bounds[y] - bounds[z]
        if delta < 0:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x + 1, z, z)  # path compression
        if h[x] > x:
            w = path_max(h, h[x])
            domains[max_sorted_vars[i], MIN] = bounds[w]
            path_set(h, x, w, w)  # path compression
        if delta == 0:
            j1 = j - 1
            path_set(h, h[y], j1, y)  # mark hall interval
            h[y] = j1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True)
def filter_upper(
    n: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
) -> bool:
    for i in range(nb + 1):
        i1 = i + 1
        t[i] = h[i] = i1
        d[i] = bounds[i1] - bounds[i]
    for i in range(n - 1, -1, -1):
        x = ranks[min_sorted_vars[i], MAX]
        y = ranks[min_sorted_vars[i], MIN]
        z = path_min(t, x - 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z - 1
            z = path_min(t, t[z])
            t[z] = j
        delta = d[z] + bounds[z] - bounds[y]
        if delta < 0:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x - 1, z, z)  # path compression
        if h[x] < x:
            w = path_min(h, h[x])
            domains[min_sorted_vars[i], MAX] = bounds[w] - 1
            path_set(h, x, w, w)  # path compression
        if delta == 0:
            j1 = j + 1
            path_set(h, h[y], j1, y)  # mark hall interval
            h[y] = j1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True, inline="always")
def argsort_into(sorted_vars: NDArray, domains: NDArray, bound: int) -> None:
    """
    Sorts the variables by their given bound, writing the permutation into sorted_vars.

    Below SORT_MAX_N, an insertion sort on the preallocated int32 output beats np.argsort, whose fixed cost
    (allocating an int64 result, copying the strided bound column, then narrowing to int32) dominates at small n.
    Inlined: as a separately cached function it would add a per-process load cost to every solver run.

    :param sorted_vars: the array of variables to sort, modified in place
    :type sorted_vars: NDArray
    :param domains: the domains of the variables
    :type domains: NDArray
    :param bound: MIN or MAX, the bound to sort on
    :type bound: int
    """
    n = len(sorted_vars)
    if n > SORT_MAX_N:
        sorted_vars[:] = np.argsort(domains[:, bound])
        return
    for i in range(n):
        sorted_vars[i] = i
    for i in range(1, n):
        var = sorted_vars[i]
        value = domains[var, bound]
        j = i - 1
        while j >= 0 and domains[sorted_vars[j], bound] > value:
            sorted_vars[j + 1] = sorted_vars[j]
            j -= 1
        sorted_vars[j + 1] = var


@njit(cache=True)
def compute_domains_alldifferent(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces that :math:`x_i <> x_j when i<>j`.

    Adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".
    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: either empty or offsets
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    has_offsets = len(parameters) != 0
    if has_offsets:
        offsets = parameters[:, np.newaxis]
        domains += offsets
    bounds_nb = 2 * (n + 1)
    empty_buffer = np.empty(4 * bounds_nb + 4 * n, dtype=np.int32)  # single allocation for all the scratch arrays
    bounds = empty_buffer[:bounds_nb]
    t = empty_buffer[bounds_nb : 2 * bounds_nb]  # critical capacity pointers
    d = empty_buffer[2 * bounds_nb : 3 * bounds_nb]  # differences between critical capacities
    h = empty_buffer[3 * bounds_nb : 4 * bounds_nb]  # Hall interval pointers
    min_sorted_vars = empty_buffer[4 * bounds_nb : 4 * bounds_nb + n]
    max_sorted_vars = empty_buffer[4 * bounds_nb + n : 4 * bounds_nb + 2 * n]
    ranks = empty_buffer[4 * bounds_nb + 2 * n :].reshape(n, 2)
    argsort_into(min_sorted_vars, domains, MIN)
    argsort_into(max_sorted_vars, domains, MAX)
    ground = True
    for i in range(n):
        if domains[i, MIN] != domains[i, MAX]:
            ground = False
            break
    nb = update_bounds(bounds, n, domains, ranks, min_sorted_vars, max_sorted_vars)
    if filter_lower(n, nb, t, d, h, bounds, domains, ranks, max_sorted_vars) and filter_upper(
        n, nb, t, d, h, bounds, domains, ranks, min_sorted_vars
    ):
        if has_offsets:
            domains -= offsets
        # all the variables were ground and pairwise distinct: the constraint stays true in the subtree
        return PROP_ENTAILMENT if ground else PROP_CONSISTENCY
    else:
        return PROP_INCONSISTENCY
