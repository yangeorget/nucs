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
        self.bounds = [0] * (2 * self.size + 2)
        self.critical_capacity_ptrs = [0] * (2 * self.size + 2)
        self.cc_diff_ptrs = [0] * (2 * self.size + 2)
        self.hall_interval_ptrs = [0] * (2 * self.size + 2)
        self.rank_domains = np.zeros((self.size, 4), dtype=int)

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        self.rank_domains[:, [MIN, MAX]] = domains.copy()
        min_sorted_vars = np.argsort(self.rank_domains[:, MIN])
        max_sorted_vars = np.argsort(self.rank_domains[:, MAX])
        # TODO: optimize
        min = self.rank_domains[min_sorted_vars[0], MIN]
        max = self.rank_domains[max_sorted_vars[0], MAX] + 1
        self.bounds[0] = last = min - 2
        i = j = nb = 0
        while True:
            if i < self.size and min <= max:
                if min != last:
                    nb += 1
                    self.bounds[nb] = last = min
                self.rank_domains[min_sorted_vars[i], MIN_RANK] = nb
                i += 1
                if i < self.size:
                    min = self.rank_domains[min_sorted_vars[i], MIN]
            else:
                if max != last:
                    nb += 1
                    self.bounds[nb] = last = max
                self.rank_domains[max_sorted_vars[j], MAX_RANK] = nb
                j += 1
                if j == self.size:
                    break
                max = self.rank_domains[max_sorted_vars[j], MAX] + 1
        self.bounds[nb + 1] = self.bounds[nb] + 2
        if not self.filter_lower(
            nb,
            self.critical_capacity_ptrs,
            self.cc_diff_ptrs,
            self.hall_interval_ptrs,
            self.bounds,
            self.rank_domains,
            max_sorted_vars,
        ):
            return None
        if not self.filter_upper(
            nb,
            self.critical_capacity_ptrs,
            self.cc_diff_ptrs,
            self.hall_interval_ptrs,
            self.bounds,
            self.rank_domains,
            min_sorted_vars,
        ):
            return None
        return self.rank_domains[:, [MIN, MAX]]

    def filter_lower(
        self,
        nb: int,
        critical_capacity_ptrs: List[int],
        cc_diff_ptrs: List[int],
        hall_interval_ptrs: List[int],
        bounds: List[int],
        rank_domains: NDArray,
        max_sorted_vars: NDArray,
    ) -> bool:
        for i in range(0, nb + 1):
            critical_capacity_ptrs[i + 1] = hall_interval_ptrs[i + 1] = i
            cc_diff_ptrs[i + 1] = bounds[i + 1] - bounds[i]
        for i in range(0, self.size):
            x = rank_domains[max_sorted_vars[i], MIN_RANK]
            y = rank_domains[max_sorted_vars[i], MAX_RANK]
            z = self.path_max(critical_capacity_ptrs, x + 1)
            j = critical_capacity_ptrs[z]
            cc_diff_ptrs[z] -= 1
            if cc_diff_ptrs[z] == 0:
                critical_capacity_ptrs[z] = z + 1
                z = self.path_max(critical_capacity_ptrs, critical_capacity_ptrs[z])
                critical_capacity_ptrs[z] = j
            self.path_set(critical_capacity_ptrs, x + 1, z, z)  # path compression
            if cc_diff_ptrs[z] < bounds[z] - bounds[y]:
                return False
            if hall_interval_ptrs[x] > x:
                w = self.path_max(hall_interval_ptrs, hall_interval_ptrs[x])
                rank_domains[max_sorted_vars[i], MIN] = bounds[w]
                self.path_set(hall_interval_ptrs, x, w, w)  # path compression
            if cc_diff_ptrs[z] == bounds[z] - bounds[y]:
                self.path_set(hall_interval_ptrs, hall_interval_ptrs[y], j - 1, y)  # mark hall interval
                hall_interval_ptrs[y] = j - 1  # hall interval[bounds[j], bounds[y]]
        return True

    def filter_upper(
        self,
        nb: int,
        critical_capacity_ptrs: List[int],
        cc_diff_ptrs: List[int],
        hall_interval_ptrs: List[int],
        bounds: List[int],
        rank_domains: NDArray,
        min_sorted_vars: NDArray,
    ) -> bool:
        for i in range(0, nb + 1):
            critical_capacity_ptrs[i] = hall_interval_ptrs[i] = i + 1
            cc_diff_ptrs[i] = bounds[i + 1] - bounds[i]
        for i in range(self.size - 1, -1, -1):
            x = rank_domains[min_sorted_vars[i], MAX_RANK]
            y = rank_domains[min_sorted_vars[i], MIN_RANK]
            z = self.path_min(critical_capacity_ptrs, x - 1)
            j = critical_capacity_ptrs[z]
            cc_diff_ptrs[z] -= 1
            if cc_diff_ptrs[z] == 0:
                critical_capacity_ptrs[z] = z - 1
                z = self.path_min(critical_capacity_ptrs, critical_capacity_ptrs[z])
                critical_capacity_ptrs[z] = j
            self.path_set(critical_capacity_ptrs, x - 1, z, z)  # path compression
            if cc_diff_ptrs[z] < bounds[y] - bounds[z]:
                return False
            if hall_interval_ptrs[x] < x:
                w = self.path_min(hall_interval_ptrs, hall_interval_ptrs[x])
                rank_domains[min_sorted_vars[i], MAX] = bounds[w] - 1
                self.path_set(hall_interval_ptrs, x, w, w)  # path compression
            if cc_diff_ptrs[z] == bounds[y] - bounds[z]:
                self.path_set(hall_interval_ptrs, hall_interval_ptrs[y], j + 1, y)  # mark hall interval
                hall_interval_ptrs[y] = j + 1  # hall interval[bounds[j], bounds[y]]
        return True

    def path_set(self, t: List[int], start: int, end: int, to: int) -> None:
        p = start
        while p != end:
            tmp = t[p]
            t[p] = to
            p = tmp

    def path_min(self, t: List[int], i: int) -> int:
        while t[i] < i:
            i = t[i]
        return i

    def path_max(self, t: List[int], i: int) -> int:
        while t[i] > i:
            i = t[i]
        return i
