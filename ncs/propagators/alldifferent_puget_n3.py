from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problem import MAX, MIN
from ncs.propagators.propagator import Propagator


class AlldifferentPugetN3(Propagator):
    """
    JF Puget's bound consistency algorithm for the alldifferent constraint.
    """

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        new_domains = domains.copy()
        if not self.update_new_domains_min(new_domains):
            return None
        if not self.update_new_domains_max(new_domains):
            return None
        return new_domains

    def update_new_domains_min(self, domains: NDArray) -> bool:
        sorted_vars = np.argsort(domains[:, MIN])
        mins = domains[sorted_vars, MIN]
        maxs = domains[sorted_vars, MAX]
        u = np.zeros(self.variables.size, dtype=int)
        for i in range(0, self.variables.size):
            if not self.insert_min(i, mins, maxs, u, domains, sorted_vars):
                return False
        return True

    def update_new_domains_max(self, domains: NDArray) -> bool:
        # TODO
        return True

    def insert_min(
        self, i: int, mins: NDArray, maxs: NDArray, u: NDArray, domains: NDArray, sorted_vars: NDArray
    ) -> bool:
        u[i] = mins[i]
        for j in range(0, i):
            if mins[j] < mins[i]:
                u[j] += 1
                if u[j] > maxs[i]:
                    return False
                if u[j] == maxs[i]:
                    self.increment_min(mins, mins[j], maxs[i], i, domains, sorted_vars)
            else:
                u[i] += 1
        if u[i] > maxs[i]:
            return False
        if u[i] == maxs[i]:
            self.increment_min(mins, mins[i], maxs[i], i, domains, sorted_vars)
        return True

    def increment_min(self, mins: NDArray, a: int, b: int, i: int, domains: NDArray, sorted_vars: NDArray) -> None:
        for j in range(i + 1, self.variables.size):
            if mins[j] >= a:
                domains[sorted_vars[j], MIN] = np.max(domains[sorted_vars[j], MIN], b + 1)

    def __str__(self) -> str:
        return f"alldifferent_puget_n3({self.variables})"
