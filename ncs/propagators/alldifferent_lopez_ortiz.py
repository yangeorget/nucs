from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.propagator import Propagator

MIN_RANK = 2
MAX_RANK = 3


class AlldifferentLopezOrtiz(Propagator):
    def __init__(self, variables: List[int]):
        super().__init__(variables)

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        self.nb = 0
        self.bounds = [0] * (2 * self.size + 2)
        self.t = [0] * (2 * self.size + 2)
        self.d = [0] * (2 * self.size + 2)
        self.h = [0] * (2 * self.size + 2)
        self.iv = np.zeros((self.size, 4), dtype=int)
        self.iv[:, MIN] = domains[:, MIN].copy()
        self.iv[:, MAX] = domains[:, MAX].copy()
        self.min_sorted_vars = np.argsort(self.iv[:, MIN])
        self.max_sorted_vars = np.argsort(self.iv[:, MAX])
        # sort
        min = self.iv[self.min_sorted_vars[0], MIN]
        max = self.iv[self.max_sorted_vars[0], MAX] + 1
        last = min - 2
        self.bounds[0] = last
        i = j = nb = 0
        while True:
            if i < self.size and min <= max:
                if min != last:
                    nb += 1
                    last = min
                    self.bounds[nb] = last
                self.iv[self.min_sorted_vars[i], MIN_RANK] = nb
                i += 1
                if i < self.size:
                    min = self.iv[self.min_sorted_vars[i], MIN]
            else:
                if max != last:
                    nb += 1
                    last = max
                    self.bounds[nb] = last
                self.iv[self.max_sorted_vars[j], MAX_RANK] = nb
                j += 1
                if j == self.size:
                    break
                max = self.iv[self.max_sorted_vars[j], MAX] + 1
        self.nb = nb
        self.bounds[nb + 1] = self.bounds[nb] + 2
        self.filter_lower()
        self.filter_upper()
        return self.iv[:, [MIN, MAX]]

    def filter_lower(self) -> bool:
        for i in range(0, self.nb + 1):
            self.t[i + 1] = self.h[i + 1] = i
            self.d[i + 1] = self.bounds[i + 1] - self.bounds[i]
        for i in range(0, self.size):
            x = self.iv[self.max_sorted_vars[i], MIN_RANK]
            y = self.iv[self.max_sorted_vars[i], MAX_RANK]
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
                self.iv[self.max_sorted_vars[i], MIN] = self.bounds[w]
                self.path_set(self.h, x, w, w)  # path compression
            if self.d[z] == self.bounds[z] - self.bounds[y]:
                self.path_set(self.h, self.h[y], j - 1, y)  # mark hall interval
                self.h[y] = j - 1  # hall interval[bounds[j], bounds[y]]
        return True

    def filter_upper(self) -> bool:
        for i in range(0, self.nb + 1):
            self.t[i] = self.h[i] = i + 1
            self.d[i] = self.bounds[i + 1] - self.bounds[i]
        for i in range(self.size - 1, -1, -1):
            x = self.iv[self.min_sorted_vars[i], MAX_RANK]
            y = self.iv[self.min_sorted_vars[i], MIN_RANK]
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
                self.iv[self.min_sorted_vars[i], MAX] = self.bounds[w] - 1
                self.path_set(self.h, x, w, w)  # path compression
            if self.d[z] == self.bounds[y] - self.bounds[z]:
                self.path_set(self.h, self.h[y], j + 1, y)  # mark hall interval
                self.h[y] = j + 1  # hall interval[bounds[j], bounds[y]]
        return True

    def path_set(self, t: List[int], start: int, end: int, to: int) -> None:
        n = start
        p = n
        while p != end:
            n = t[p]
            t[p] = to
            p = n

    def path_min(self, t: List[int], i: int) -> int:
        while t[i] < i:
            i = t[i]
        return i

    def path_max(self, t: List[int], i: int) -> int:
        while t[i] > i:
            i = t[i]
        return i
