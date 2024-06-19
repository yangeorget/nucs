from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.propagator import Propagator


class AlldifferentPugetN2(Propagator):
    """
    JF Puget's bound consistency algorithm for the alldifferent constraint with quadratic complexity.
    """

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        new_domains = domains.copy()
        if not self.update_mins(new_domains):
            return None
        new_domains[:, [MIN, MAX]] = -new_domains[:, [MAX, MIN]]
        if not self.update_mins(new_domains):
            return None
        new_domains[:, [MIN, MAX]] = -new_domains[:, [MAX, MIN]]
        return new_domains

    def update_mins(self, new_domains: NDArray) -> bool:
        sorted_vars = np.argsort(new_domains[:, MAX])
        sorted_domains = new_domains[sorted_vars]
        u = np.zeros(self.size, dtype=int)
        for i in range(0, self.size):
            if not self.insert(new_domains, i, u, sorted_domains, sorted_vars):
                return False
        return True

    def insert(self, new_doms: NDArray, i: int, u: NDArray, sorted_doms: NDArray, sorted_vars: NDArray) -> bool:
        sorted_doms_i_min = sorted_doms[i, MIN]
        sorted_doms_i_max = sorted_doms[i, MAX]
        u[i] = sorted_doms_i_min
        best_min = self.size + 1
        for j in range(0, i):
            sorted_doms_j_min = sorted_doms[j, MIN]
            if sorted_doms_j_min < sorted_doms_i_min:
                u[j] += 1
                if u[j] > sorted_doms_i_max:
                    return False
                if u[j] == sorted_doms_i_max and sorted_doms_j_min < best_min:
                    best_min = sorted_doms_j_min
            else:
                u[i] += 1
        if u[i] > sorted_doms_i_max:
            return False
        if u[i] == sorted_doms_i_max and sorted_doms_i_min < best_min:
            best_min = sorted_doms_i_min
        if best_min <= self.size:
            self.increment_min(new_doms, i, best_min, sorted_doms_i_max, sorted_doms, sorted_vars)
        return True

    def increment_min(
        self, new_doms: NDArray, i: int, a: int, b: int, sorted_doms: NDArray, sorted_vars: NDArray
    ) -> None:
        for j in range(i + 1, self.size):
            sorted_doms_j_min = sorted_doms[j, MIN]
            if sorted_doms_j_min >= a:
                new_doms[sorted_vars[j], MIN] = max(b + 1, sorted_doms_j_min)
