from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.propagator import Propagator


class AlldifferentPugetN3(Propagator):
    """
    JF Puget's bound consistency algorithm for the alldifferent constraint with cubic complexity.
    """

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        new_domains = domains[self.variables]
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
        u[i] = sorted_doms[i, MIN]
        for j in range(0, i):
            if sorted_doms[j, MIN] < sorted_doms[i, MIN]:
                u[j] += 1
                if u[j] > sorted_doms[i, MAX]:
                    return False
                if u[j] == sorted_doms[i, MAX]:
                    self.increment_min(new_doms, i, sorted_doms[j, MIN], sorted_doms[i, MAX], sorted_doms, sorted_vars)
            else:
                u[i] += 1
        if u[i] > sorted_doms[i, MAX]:
            return False
        if u[i] == sorted_doms[i, MAX]:
            self.increment_min(new_doms, i, sorted_doms[i, MIN], sorted_doms[i, MAX], sorted_doms, sorted_vars)
        return True

    def increment_min(
        self, new_doms: NDArray, i: int, a: int, b: int, sorted_doms: NDArray, sorted_vars: NDArray
    ) -> None:
        for j in range(i + 1, self.size):
            if sorted_doms[j, MIN] >= a:
                new_doms[sorted_vars[j], MIN] = max(b + 1, sorted_doms[j, MIN])

    def __str__(self) -> str:
        return f"alldifferent_puget_n3({self.variables})"
