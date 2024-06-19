from typing import Optional, List
from numba import jit

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.propagator import Propagator

MIN_RANK = 2
MAX_RANK = 3

@jit(nopython=True)
def compute_nb(
    size, rank_domains: NDArray, min_sorted_vars: NDArray, max_sorted_vars: NDArray, bounds: List[int]
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

@jit(nopython=True)
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
    for i in range(0, nb + 1):
        t[i + 1] = h[i + 1] = i
        d[i + 1] = bounds[i + 1] - bounds[i]
    for i in range(0, size):
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
        path_set(t, x + 1, z, z)  # path compression
        if d[z] < bounds[z] - bounds[y]:
            return False
        if h[x] > x:
            w = path_max(h, h[x])
            rank_domains[max_sorted_vars_i, MIN] = bounds[w]
            path_set(h, x, w, w)  # path compression
        if d[z] == bounds[z] - bounds[y]:
            path_set(h, h[y], j - 1, y)  # mark hall interval
            h[y] = j - 1  # hall interval[bounds[j], bounds[y]]
    return True

@jit(nopython=True)
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
    for i in range(0, nb + 1):
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
        path_set(t, x - 1, z, z)  # path compression
        if d[z] < bounds[y] - bounds[z]:
            return False
        if h[x] < x:
            w = path_min(h, h[x])
            rank_domains[min_sorted_vars_i, MAX] = bounds[w] - 1
            path_set(h, x, w, w)  # path compression
        if d[z] == bounds[y] - bounds[z]:
            path_set(h, h[y], j + 1, y)  # mark hall interval
            h[y] = j + 1  # hall interval[bounds[j], bounds[y]]
    return True

@jit(nopython=True)
def path_set(t: NDArray, start: int, end: int, to: int) -> None:
    p = start
    while p != end:
        tmp = t[p]
        t[p] = to
        p = tmp

@jit(nopython=True)
def path_min(t: NDArray, i: int) -> int:
    while t[i] < i:
        i = t[i]
    return i

@jit(nopython=True)
def path_max(t: NDArray, i: int) -> int:
    while t[i] > i:
        i = t[i]
    return i


# TODO: compile
def compute_domains(size: int, domains: NDArray) -> Optional[NDArray]:
    bound_len = 2 * size + 2
    bounds = np.zeros(bound_len, dtype=int)
    t = np.zeros(bound_len, dtype=int)
    d = np.zeros(bound_len, dtype=int)
    h = np.zeros(bound_len, dtype=int)
    rank_domains = np.zeros((size, 4), dtype=int)
    rank_domains[:, [MIN, MAX]] = domains.copy()
    min_sorted_vars = np.argsort(rank_domains[:, MIN])
    max_sorted_vars = np.argsort(rank_domains[:, MAX])
    nb = compute_nb(size, rank_domains, min_sorted_vars, max_sorted_vars, bounds)
    if not filter_lower(size,nb, t, d, h, bounds, rank_domains, max_sorted_vars):
        return None
    if not filter_upper(size,nb, t, d, h, bounds, rank_domains, min_sorted_vars):
        return None
    return rank_domains[:, [MIN, MAX]]

class AlldifferentLopezOrtiz(Propagator):
    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        return compute_domains(self.size, domains)

