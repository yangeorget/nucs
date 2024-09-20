import math

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_alldifferent(n: int, data: NDArray) -> float:
    return 2 * n * math.log(n) + 5 * n


def get_triggers_alldifferent(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def path_set(t: NDArray, start: int, end: int, to: int) -> None:
    while (p := start) != end:
        start = t[p]
        t[p] = to


@njit(cache=True)
def path_min(t: NDArray, i: int) -> int:
    while t[i] < i:
        i = t[i]
    return i


@njit(cache=True)
def path_max(t: NDArray, i: int) -> int:
    while t[i] > i:
        i = t[i]
    return i


@njit(cache=True)
def compute_nb(
    n: int,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    max_sorted_vars: NDArray,
    bounds: NDArray,
) -> int:
    min_value = domains[min_sorted_vars[0], MIN]
    max_value = domains[max_sorted_vars[0], MAX] + 1
    bounds[0] = last = min_value - 2
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
        t[i] = h[i] = i - 1
        d[i] = bounds[i] - bounds[i - 1]
    for i, max_sorted_vars_i in enumerate(max_sorted_vars):
        x = ranks[max_sorted_vars_i, MIN]
        y = ranks[max_sorted_vars_i, MAX]
        z = path_max(t, x + 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z + 1
            z = path_max(t, t[z])
            t[z] = j
        if d[z] + bounds[y] < bounds[z]:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x + 1, z, z)  # path compression
        if h[x] > x:
            w = path_max(h, h[x])
            domains[max_sorted_vars_i, MIN] = bounds[w]
            path_set(h, x, w, w)  # path compression
        if d[z] + bounds[y] == bounds[z]:
            path_set(h, h[y], j - 1, y)  # mark hall interval
            h[y] = j - 1  # hall interval[bounds[j], bounds[y]]
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
        t[i] = h[i] = i + 1
        d[i] = bounds[i + 1] - bounds[i]
    for i in range(n - 1, -1, -1):
        min_sorted_vars_i = min_sorted_vars[i]
        x = ranks[min_sorted_vars_i, MAX]
        y = ranks[min_sorted_vars_i, MIN]
        z = path_min(t, x - 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z - 1
            z = path_min(t, t[z])
            t[z] = j
        if d[z] + bounds[z] < bounds[y]:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x - 1, z, z)  # path compression
        if h[x] < x:
            w = path_min(h, h[x])
            domains[min_sorted_vars_i, MAX] = bounds[w] - 1
            path_set(h, x, w, w)  # path compression
        if d[z] + bounds[z] == bounds[y]:
            path_set(h, h[y], j + 1, y)  # mark hall interval
            h[y] = j + 1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True)
def compute_domains_alldifferent(domains: NDArray, data: NDArray) -> int:
    """
    Adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".
    :param domains: the domains of the variables
    :param data: unused here
    """
    n = len(domains)
    ranks = np.zeros((n, 2), dtype=np.uint16)
    bounds_nb = 2 * n + 2
    bounds = np.zeros(bounds_nb, dtype=np.int32)
    min_sorted_vars = np.argsort(domains[:, MIN])
    max_sorted_vars = np.argsort(domains[:, MAX])
    nb = compute_nb(n, domains, ranks, min_sorted_vars, max_sorted_vars, bounds)
    t = np.zeros(bounds_nb, dtype=np.uint16)  # critical capacity pointers
    d = np.zeros(bounds_nb, dtype=np.int32)  # differences between critical capacities
    h = np.zeros(bounds_nb, dtype=np.uint16)  # Hall interval pointers
    return (
        PROP_CONSISTENCY
        if filter_lower(n, nb, t, d, h, bounds, domains, ranks, max_sorted_vars)
        and filter_upper(n, nb, t, d, h, bounds, domains, ranks, min_sorted_vars)
        else PROP_INCONSISTENCY
    )
