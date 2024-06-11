from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.propagator import Propagator


class AlldifferentLopezOrtiz(Propagator):
    def __init__(self, variables: List[int]):
        super().__init__(variables)
        self.nb = 0
        self.bounds = [0] * (2 * self.size + 2)
        self.t = [0] * (2 * self.size + 2)
        self.d = [0] * (2 * self.size + 2)
        self.h = [0] * (2 * self.size + 2)
        self.min_sorted_vars = np.zeros(self.size, dtype=int)
        self.max_sorted_vars = np.zeros(self.size, dtype=int)
        self.min_sorted_ranks = np.zeros((self.size, 2), dtype=int)
        self.max_sorted_ranks = np.zeros((self.size, 2), dtype=int)

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        iv_domains = domains.copy()
        self.min_sorted_vars = np.argsort(iv_domains[:, MIN])
        self.max_sorted_vars = np.argsort(iv_domains[:, MAX])
        # sort
        min = iv_domains[self.min_sorted_vars[0], MIN]
        max = iv_domains[self.max_sorted_vars[0], MAX] + 1
        self.bounds[0] = last = min - 2
        i = j = nb = 0
        while True:
            if i < self.size and min <= max:
                if min != last:
                    nb += 1
                    self.bounds[nb] = last = min
                self.min_sorted_ranks[i, MIN] = nb
                i += 1
                if i < self.size:
                    min = iv_domains[self.min_sorted_vars[i], MIN]
            else:
                if max != last:
                    nb += 1
                    self.bounds[nb] = last = max
                self.max_sorted_ranks[j, MAX] = nb
                j += 1
                if j == self.size:
                    break
                max = iv_domains[self.max_sorted_vars[j], MAX] + 1
        self.nb = nb
        self.bounds[nb+1] = self.bounds[nb] +2
        self.filter_lower(iv_domains)
        self.filter_upper(iv_domains)
        return iv_domains

    def filter_lower(self, iv_domains: NDArray) -> bool:
        for i in range(0, self.nb + 1):
            self.t[i + 1] = self.h[i + 1] = i
            self.d[i + 1] = self.bounds[i + 1] - self.bounds[i]
        for i in range(0, self.size):
            x = self.max_sorted_ranks[i, MIN]
            y = self.max_sorted_ranks[i, MAX]
            z = self.path_max(self.t, x + 1)
            j = self.t[z]
            self.d[z] -= 1
            if self.d[z] == 0:
                self.t[z] = z + 1
                z = self.path_max(self.t, self.t[z])
                self.t[z] = j
            self.path_set(self.t, x + 1, z, z)  # path compression
            if self.d[z] < self.bounds[z] - self.bounds[y]:
                return False
            if self.h[x] > x:
                w = self.path_max(self.h, self.h[x])
                iv_domains[self.max_sorted_vars[i], MIN] = self.bounds[w]
                self.path_set(self.h, x, w, w)  # path compression
            if self.d[z] == self.bounds[z] - self.bounds[y]:
                self.path_set(self.h, self.h[y], j - 1, y)  # mark hall interval
                self.h[y] = j - 1  # hall interval[bounds[j], bounds[y]]
        return True

    def filter_upper(self, iv_domains: NDArray) -> bool:
        for i in range(0, self.nb + 1):
            self.t[i] = self.h[i] = i + 1
            self.d[i] = self.bounds[i + 1] - self.bounds[i]
        for i in range(self.size-1, -1, -1):
            x = self.min_sorted_ranks[i, MAX]
            y = self.min_sorted_ranks[i, MIN]
            z = self.path_min(self.t, x - 1)
            j = self.t[z]
            self.d[z] -= 1
            if self.d[z] == 0:
                self.t[z] = z - 1
                z = self.path_min(self.t, self.t[z])
                self.t[z] = j
            self.path_set(self.t, x - 1, z, z)  # path compression
            if self.d[z] < self.bounds[y] - self.bounds[z]:
                return False
            if self.h[x] < x:
                w = self.path_min(self.h, self.h[x])
                iv_domains[self.min_sorted_vars[i], MAX] = self.bounds[w] - 1
                self.path_set(self.h, x, w, w)  # path compression
            if self.d[z] == self.bounds[y] - self.bounds[z]:
                self.path_set(self.h, self.h[y], j + 1, y)  # mark hall interval
                self.h[y] = j + 1  # hall interval[bounds[j], bounds[y]]
        return True

    def path_set(self, t: List[int], start: int, end: int, to: int) -> None:
        k = start
        l = start
        while k != end:
            l = t[k]
            t[k] = to
            k = l

    def path_min(self, t: List[int], i: int) -> int:
        while t[i] < i:
            i = t[i]
        return i

    def path_max(self, t: List[int], i: int) -> int:
        while t[i] > i:
            i = t[i]
        return i
