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
from collections.abc import Sequence

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.propagators.alldifferent_propagator import argsort_into, path_max, path_min, path_set


def get_complexity_gcc(n: int, parameters: NDArray) -> int:
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
def get_triggers_gcc(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param n: the number of variables
    :type n: int
    :param variable: the index of the variable
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray
    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def init_partial_sum(first_value: int, m: int, values: NDArray) -> NDArray:
    """
    Inits the partial_sum data structure:
    ---------------------
    | sm | first_value |
    ---------------------
    | ds  | last_value  |
    ---------------------
    """
    partial_sum = np.zeros((2, m + 6), dtype=np.int32)
    partial_sum[0, -1] = first_value - 3
    partial_sum[1, -1] = first_value + m + 1
    sm = partial_sum[0, :-1]
    sm[0] = 0
    sm[1] = 1
    sm[2] = 2
    for i in range(2, m + 2):
        sm[i + 1] = sm[i] + values[i - 2]
    sm[m + 3] = sm[m + 2] + 1
    sm[m + 4] = sm[m + 3] + 1
    ds = partial_sum[1, :-1]
    i = m + 3
    j = m + 4
    while i > 0:
        while sm[i] == sm[i - 1]:
            ds[i] = j
            i -= 1
        ds[j] = i
        j = i
        i -= 1
    ds[j] = 0
    return partial_sum


@njit(cache=True)
def get_sum(psum: NDArray, start: int, end: int) -> int:
    fv = psum[0, -1]
    sum = psum[0, :-1]
    if start <= end:
        # assert fv <= start
        # assert end <= get_last_value(psum)
        return sum[end - fv] - sum[start - fv - 1]
    else:
        # assert fv <= end
        # assert start <= get_last_value(psum)
        return sum[end - fv - 1] - sum[start - fv]


@njit(cache=True)
def get_min_value(psum: NDArray) -> int:
    return psum[0, -1] + 3


@njit(cache=True)
def get_max_value(psum: NDArray) -> int:
    return psum[1, -1] - 2


@njit(cache=True)
def skip_non_null_elements_right(psum: NDArray, value: int) -> int:
    value -= psum[0, -1]
    ps = psum[1, value]
    return (max(ps, value)) + psum[0, -1]


@njit(cache=True)
def skip_non_null_elements_left(psum: NDArray, value: int) -> int:
    value -= psum[0, -1]
    ps = psum[1, value]
    return (psum[1, ps] if ps > value else value) + psum[0, -1]


@njit(cache=True)
def update_bounds(
    bounds: NDArray,
    n: int,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    max_sorted_vars: NDArray,
    l: NDArray,
    u: NDArray,
) -> int:
    min_value = domains[min_sorted_vars[0], DOMAIN_MIN]
    max_value = domains[max_sorted_vars[0], DOMAIN_MAX] + 1
    bounds[0] = last = l[0, -1] + 1
    i = j = nb = 0
    while True:
        if i < n and min_value <= max_value:
            if min_value != last:
                nb += 1
                bounds[nb] = last = min_value
            ranks[min_sorted_vars[i], DOMAIN_MIN] = nb
            i += 1
            if i < n:
                min_value = domains[min_sorted_vars[i], DOMAIN_MIN]
        else:
            if max_value != last:
                nb += 1
                bounds[nb] = last = max_value
            ranks[max_sorted_vars[j], DOMAIN_MAX] = nb
            j += 1
            if j == n:
                break
            max_value = domains[max_sorted_vars[j], DOMAIN_MAX] + 1
    bounds[nb + 1] = u[1, -1] + 1
    return nb


@njit(cache=True)
def filter_lower_max(
    n: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    max_sorted_vars: NDArray,
    u: NDArray,
) -> bool:
    for i in range(1, nb + 2):
        i1 = i - 1
        t[i] = h[i] = i1
        d[i] = get_sum(u, bounds[i1], bounds[i] - 1)
    for i in range(len(max_sorted_vars)):
        x = ranks[max_sorted_vars[i], DOMAIN_MIN]
        y = ranks[max_sorted_vars[i], DOMAIN_MAX]
        z = path_max(t, x + 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z + 1
            z = path_max(t, t[z])
            t[z] = j
        delta = d[z] - get_sum(u, bounds[y], bounds[z] - 1)
        if delta < 0:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x + 1, z, z)  # path compression
        if h[x] > x:
            w = path_max(h, h[x])
            domains[max_sorted_vars[i], DOMAIN_MIN] = bounds[w]
            path_set(h, x, w, w)  # path compression
            # changes = 1
        if delta == 0:
            j1 = j - 1
            path_set(h, h[y], j1, y)  # mark hall interval
            h[y] = j1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True)
def filter_upper_max(
    n: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    u: NDArray,
) -> bool:
    for i in range(nb + 1):
        i1 = i + 1
        t[i] = h[i] = i1
        d[i] = get_sum(u, bounds[i], bounds[i1] - 1)
    for i in range(n - 1, -1, -1):
        x = ranks[min_sorted_vars[i], DOMAIN_MAX]
        y = ranks[min_sorted_vars[i], DOMAIN_MIN]
        z = path_min(t, x - 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z - 1
            z = path_min(t, t[z])
            t[z] = j
        delta = d[z] - get_sum(u, bounds[z], bounds[y] - 1)
        if delta < 0:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x - 1, z, z)  # path compression
        if h[x] < x:
            w = path_min(h, h[x])
            domains[min_sorted_vars[i], DOMAIN_MAX] = bounds[w] - 1
            path_set(h, x, w, w)  # path compression
            # changes = 1
        if delta == 0:
            j1 = j + 1
            path_set(h, h[y], j1, y)  # mark hall interval
            h[y] = j1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True)
def filter_lower_min(
    n: int,
    nb: int,
    tl: NDArray,
    c: NDArray,
    sets: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    max_sorted_vars: NDArray,
    l: NDArray,
    stable_intervals: NDArray,
    stable_sets: NDArray,
    new_mins: NDArray,
) -> bool:
    w = nb + 1
    for i in range(nb + 1, 0, -1):
        stable_sets[i] = stable_intervals[i] = i - 1
        c[i] = get_sum(l, bounds[i - 1], bounds[i] - 1)
        if c[i] == 0:  # if the capacity between both bounds is zero, we have an unstable set between these two bounds
            sets[i - 1] = w
        else:
            sets[w] = i - 1
            w = i - 1
    w = nb + 1
    for i in range(nb + 1, -1, -1):
        if c[i] == 0:
            tl[i] = w
        else:
            tl[w] = w = i
    for i in range(len(max_sorted_vars)):  # visit intervals in increasing max order
        x = ranks[max_sorted_vars[i], DOMAIN_MIN]
        y = ranks[max_sorted_vars[i], DOMAIN_MAX]
        z = path_max(tl, x + 1)
        j = tl[z]
        if z != x + 1:
            # If bounds[z] - 1 belongs to a stable set, [bounds[x], bounds[z]) is a sub set of this stable set.
            w = path_max(stable_sets, x + 1)
            v = stable_sets[w]
            path_set(stable_sets, x + 1, w, w)  # path compression
            w = min(y, z)
            path_set(stable_sets, stable_sets[w], v, w)
            stable_sets[w] = v
        if c[z] <= get_sum(l, bounds[y], bounds[z] - 1):
            # (potentialStableSets[y], y] is a stable set
            w = path_max(stable_intervals, stable_sets[y])
            path_set(stable_intervals, stable_sets[y], w, w)  # path compression
            v = stable_intervals[w]
            path_set(stable_intervals, stable_intervals[y], v, y)
            stable_intervals[y] = v
        else:
            c[z] -= 1  # decrease the capacity between the two bounds
            if c[z] == 0:
                tl[z] = z + 1
                z = path_max(tl, tl[z])
                tl[z] = j
            # If the lower bound belongs to an unstable or a stable set, remind the new value we might assign to
            # the lower bound in case the variable doesn't belong to a stable set.
            if sets[x] > x:
                w = path_max(sets, x)
                new_mins[i] = w
                path_set(sets, x, w, w)  # path compression
            else:
                new_mins[i] = x  # do not shrink the variable
            if c[z] == get_sum(l, bounds[y], bounds[z] - 1):  # if an unstable set is discovered
                # consider stable and unstable sets beyond y (pathmax; the path is fully compressed)
                y = max(y, sets[y])
                path_set(sets, sets[y], j - 1, y)  # mark the new unstable set
                sets[y] = j - 1
        path_set(tl, x + 1, z, z)  # path compression
    if sets[nb] != 0:  # if there is a failure set
        return False
    # Perform path compression over all elements in the stable interval data structure. This data structure will no
    # longer be modified and will be accessed n or 2n times. Therefore, we can afford a linear time compression.
    for i in range(nb + 1, 0, -1):
        if stable_intervals[i] > i:
            stable_intervals[i] = w
        else:
            w = i
    # For all variables that are not a subset of a stable set, shrink the lower bound.
    for i in range(n - 1, -1, -1):
        x = ranks[max_sorted_vars[i], DOMAIN_MIN]
        y = ranks[max_sorted_vars[i], DOMAIN_MAX]
        if stable_intervals[x] <= x or y > stable_intervals[x]:
            domains[max_sorted_vars[i], DOMAIN_MIN] = skip_non_null_elements_right(l, bounds[new_mins[i]])
            # changes = 1
    return True


@njit(cache=True)
def filter_upper_min(
    n: int,
    nb: int,
    tl: NDArray,
    c: NDArray,
    sets: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    l: NDArray,
    stable_intervals: NDArray,
    new_maxs: NDArray,
) -> bool:
    w = 0
    for i in range(nb + 1):
        c[i] = get_sum(l, bounds[i], bounds[i + 1] - 1)
        if c[i] == 0:  # if the capacity between both bounds is zero, we have an unstable set between these two bounds
            tl[i] = w
        else:
            tl[w] = w = i
    tl[w] = nb + 1
    w = 0
    for i in range(1, nb + 1):
        if c[i - 1] == 0:
            sets[i] = w
        else:
            sets[w] = i
            w = i
    sets[w] = nb + 1
    for i in range(n - 1, -1, -1):  # visit intervals in decreasing max order
        x = ranks[min_sorted_vars[i], DOMAIN_MAX]
        y = ranks[min_sorted_vars[i], DOMAIN_MIN]
        # solve the lower bound problem
        z = path_min(tl, x - 1)
        j = tl[z]
        # If the variable is not in a discovered stable set
        # Possible optimization: use the array stbl_intervals to perform this test
        if c[z] > get_sum(l, bounds[z], bounds[y] - 1):
            c[z] -= 1
            if c[z] == 0:
                tl[z] = z - 1
                z = path_min(tl, tl[z])
                tl[z] = j
            if sets[x] < x:
                w = path_min(sets, sets[x])
                new_maxs[i] = w
                path_set(sets, x, w, w)  # path compression
            else:
                new_maxs[i] = x
            if c[z] == get_sum(l, bounds[z], bounds[y] - 1):
                y = min(y, sets[y])
                path_set(sets, sets[y], j + 1, y)  # loop
                sets[y] = j + 1
        path_set(tl, x - 1, z, z)
    #  For all variables that are not subsets of a stable set, shrink the lower bound.
    for i in range(n - 1, -1, -1):
        x = ranks[min_sorted_vars[i], DOMAIN_MIN]
        y = ranks[min_sorted_vars[i], DOMAIN_MAX]
        if stable_intervals[x] <= x or y > stable_intervals[x]:
            domains[min_sorted_vars[i], DOMAIN_MAX] = skip_non_null_elements_left(l, bounds[new_maxs[i]] - 1)
            # changes = 1
    return True


def is_vacuous_gcc(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    """
    Returns whether the parameters make the constraint vacuous, whatever the domains.

    A value that no variable is required to take (lower capacity 0) and that all of them may take (upper
    capacity at least n) constrains nothing. When that holds for every value, every assignment satisfies the
    constraint, so it need never be posted. The upper capacities are scanned first because that is what fails
    fast on a constraint that does bind. MiniZinc produces these in quantity: global_cardinality_low_up leaves
    the values outside its cover unconstrained, and a cover whose own capacities do not bite makes the lot
    vacuous.

    :param n: the number of variables
    :type n: int
    :param parameters: the domain offset, then the lower capacities, then the upper capacities
    :type parameters: Sequence[int]
    :param domains: the initial domains, unused here
    :type domains: Sequence[tuple[int, int]]

    :return: True when no assignment can violate the constraint
    :rtype: bool
    """
    m = (len(parameters) - 1) >> 1  # number of values
    for j in range(m):
        if parameters[1 + m + j] < n:
            return False
    for j in range(m):
        if parameters[1 + j] != 0:
            return False
    return True


@njit(cache=True)
def compute_domains_gcc(domains: NDArray, parameters: NDArray) -> int:
    r"""
    This propagator (Global Cardinality Constraint) enforces that
    :math:`l_j \le |\{ i : x_i = v_j \}| \le c_j` for all j.
    It is adapted from "An efficient bounds consistency algorithm for the global cardinality constraint".

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: there are 1 + 2 * m parameters:
                       the first domain value (v_0), then the m lower bounds, then the m upper bounds (capacities)
    :type parameters: NDArray
    :return: a propagation status (PROP_INCONSISTENCY or PROP_CONSISTENCY)
    :rtype: int
    """
    n = len(domains)
    m = (len(parameters) - 1) >> 1  # number of values
    bounds_nb = 2 * (n + 1)
    empty_buffer = np.empty(4 * bounds_nb + 4 * n, dtype=np.int32)  # single allocation for all the scratch arrays
    bounds = empty_buffer[:bounds_nb]
    t = empty_buffer[bounds_nb : 2 * bounds_nb]  # critical capacity pointers
    d = empty_buffer[2 * bounds_nb : 3 * bounds_nb]  # differences between critical capacities
    h = empty_buffer[3 * bounds_nb : 4 * bounds_nb]  # Hall interval pointers
    min_sorted_vars = empty_buffer[4 * bounds_nb : 4 * bounds_nb + n]
    max_sorted_vars = empty_buffer[4 * bounds_nb + n : 4 * bounds_nb + 2 * n]
    ranks = empty_buffer[4 * bounds_nb + 2 * n :].reshape(n, 2)
    zero_buffer = np.zeros(2 * bounds_nb + n, dtype=np.int32)  # to reduce the number of allocations
    stable_intervals = zero_buffer[:bounds_nb]
    stable_sets = zero_buffer[bounds_nb : 2 * bounds_nb]
    new_mins = zero_buffer[2 * bounds_nb :]
    l = init_partial_sum(parameters[0], m, parameters[1 : 1 + m])
    u = init_partial_sum(parameters[0], m, parameters[1 + m :])
    argsort_into(min_sorted_vars, domains, DOMAIN_MIN)
    argsort_into(max_sorted_vars, domains, DOMAIN_MAX)
    nb = update_bounds(bounds, n, domains, ranks, min_sorted_vars, max_sorted_vars, l, u)
    # assert get_min_value(l) == get_min_value(u)
    # assert get_max_value(l) == get_max_value(u)
    # assert get_min_value(l) <= domains[min_sorted_vars[0], DOMAIN_MIN]
    # assert domains[max_sorted_vars[n - 1], DOMAIN_MAX] <= get_max_value(u)
    if get_sum(l, get_min_value(l), domains[min_sorted_vars[0], DOMAIN_MIN] - 1) > 0:
        return PROP_INCONSISTENCY
    if get_sum(l, domains[max_sorted_vars[n - 1], DOMAIN_MAX] + 1, get_max_value(l)) > 0:
        return PROP_INCONSISTENCY
    if not filter_lower_max(n, nb, t, d, h, bounds, domains, ranks, max_sorted_vars, u):
        return PROP_INCONSISTENCY
    if not filter_lower_min(
        n,
        nb,
        t,
        d,
        h,
        bounds,
        domains,
        ranks,
        max_sorted_vars,
        l,
        stable_intervals,
        stable_sets,
        new_mins,
    ):
        return PROP_INCONSISTENCY
    if not filter_upper_max(n, nb, t, d, h, bounds, domains, ranks, min_sorted_vars, u):
        return PROP_INCONSISTENCY
    if not filter_upper_min(n, nb, t, d, h, bounds, domains, ranks, min_sorted_vars, l, stable_intervals, new_mins):
        return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
