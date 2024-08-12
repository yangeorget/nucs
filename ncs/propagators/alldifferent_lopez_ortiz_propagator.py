from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.utils import MAX, MIN, init_triggers

MIN_RANK = 2
MAX_RANK = 3


def get_triggers(n: int, data: NDArray) -> NDArray:
    return init_triggers(n, True)


@jit(nopython=True, cache=True)
def compute_nb(
    size: int, rank_domains: NDArray, min_sorted_vars: NDArray, max_sorted_vars: NDArray, bounds: NDArray
) -> int:
    min = rank_domains[min_sorted_vars[0], MIN]
    max = rank_domains[max_sorted_vars[0], MAX] + 1
    bounds[0] = last = min - 2
    i = j = nb = 0
    while True:
        if i < size and min <= max:
            if min != last:
                nb += 1
                bounds[nb] = last = min
            rank_domains[min_sorted_vars[i], MIN_RANK] = nb
            i += 1
            if i < size:
                min = rank_domains[min_sorted_vars[i], MIN]
        else:
            if max != last:
                nb += 1
                bounds[nb] = last = max
            rank_domains[max_sorted_vars[j], MAX_RANK] = nb
            j += 1
            if j == size:
                break
            max = rank_domains[max_sorted_vars[j], MAX] + 1
    bounds[nb + 1] = bounds[nb] + 2
    return nb


@jit(nopython=True, cache=True)
def filter_lower(
    size: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    rank_domains: NDArray,
    max_sorted_vars: NDArray,
) -> bool:
    for i in range(1, nb + 2):
        t[i] = h[i] = i - 1
        d[i] = bounds[i] - bounds[i - 1]
    for i in range(size):
        max_sorted_vars_i = max_sorted_vars[i]
        x = rank_domains[max_sorted_vars_i, MIN_RANK]
        y = rank_domains[max_sorted_vars_i, MAX_RANK]
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
            rank_domains[max_sorted_vars_i, MIN] = bounds[w]
            path_set(h, x, w, w)  # path compression
        if d[z] + bounds[y] == bounds[z]:
            path_set(h, h[y], j - 1, y)  # mark hall interval
            h[y] = j - 1  # hall interval[bounds[j], bounds[y]]
    return True


@jit(nopython=True, cache=True)
def filter_upper(
    size: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    rank_domains: NDArray,
    min_sorted_vars: NDArray,
) -> bool:
    for i in range(nb + 1):
        t[i] = h[i] = i + 1
        d[i] = bounds[i + 1] - bounds[i]
    for i in range(size - 1, -1, -1):
        min_sorted_vars_i = min_sorted_vars[i]
        x = rank_domains[min_sorted_vars_i, MAX_RANK]
        y = rank_domains[min_sorted_vars_i, MIN_RANK]
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
            rank_domains[min_sorted_vars_i, MAX] = bounds[w] - 1
            path_set(h, x, w, w)  # path compression
        if d[z] + bounds[z] == bounds[y]:
            path_set(h, h[y], j + 1, y)  # mark hall interval
            h[y] = j + 1  # hall interval[bounds[j], bounds[y]]
    return True


@jit(nopython=True, cache=True)
def path_set(t: NDArray, start: int, end: int, to: int) -> None:
    while (p := start) != end:
        start = t[p]
        t[p] = to


@jit(nopython=True, cache=True)
def path_min(t: NDArray, i: int) -> int:
    while t[i] < i:
        i = t[i]
    return i


@jit(nopython=True, cache=True)
def path_max(t: NDArray, i: int) -> int:
    while t[i] > i:
        i = t[i]
    return i


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: Optional[NDArray] = None) -> Optional[NDArray]:
    """
    Adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    size = len(domains)
    rank_domains = np.hstack((domains, np.zeros((size, 2), dtype=np.int16)))
    bounds_nb = 2 * size + 2
    bounds = np.zeros(bounds_nb, dtype=np.int32)
    min_sorted_vars = np.argsort(rank_domains[:, MIN])
    max_sorted_vars = np.argsort(rank_domains[:, MAX])
    nb = compute_nb(size, rank_domains, min_sorted_vars, max_sorted_vars, bounds)
    t = np.zeros(bounds_nb, dtype=np.uint16)  # critical capacity pointers
    d = np.zeros(bounds_nb, dtype=np.int32)  # differences between critical capacities
    h = np.zeros(bounds_nb, dtype=np.uint16)  # Hall interval pointers
    if filter_lower(size, nb, t, d, h, bounds, rank_domains, max_sorted_vars) and filter_upper(
        size, nb, t, d, h, bounds, rank_domains, min_sorted_vars
    ):
        return rank_domains[:, :2]
    else:
        return None

