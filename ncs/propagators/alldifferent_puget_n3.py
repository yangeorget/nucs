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

    def update_new_domains_min(self, new_domains: NDArray) -> bool:
        sorted_vars = np.argsort(new_domains[:, MIN])
        sorted_domains = new_domains[sorted_vars]
        u = np.zeros(self.variables.size, dtype=int)
        for i in range(0, self.variables.size):
            if not self.insert_min(new_domains, i, u, sorted_domains, sorted_vars):
                return False
        return True

    def update_new_domains_max(self, domains: NDArray) -> bool:
        # TODO
        return True

    def insert_min(
        self, new_domains: NDArray, i: int, u: NDArray, sorted_domains: NDArray, sorted_vars: NDArray
    ) -> bool:
        u[i] = sorted_domains[i, MIN]
        for j in range(0, i):
            if sorted_domains[j, MIN] < sorted_domains[i, MIN]:
                u[j] += 1
                if u[j] > sorted_domains[i, MAX]:
                    return False
                if u[j] == sorted_domains[i, MAX]:
                    self.increment_min(
                        new_domains, i, sorted_domains[j, MIN], sorted_domains[i, MAX], sorted_domains, sorted_vars
                    )
            else:
                u[i] += 1
        if u[i] > sorted_domains[i, MAX]:
            return False
        if u[i] == sorted_domains[i, MAX]:
            self.increment_min(
                new_domains, i, sorted_domains[i, MIN], sorted_domains[i, MAX], sorted_domains, sorted_vars
            )
        return True

    def increment_min(
        self, new_domains: NDArray, i: int, a: int, b: int, sorted_domains: NDArray, sorted_vars: NDArray
    ) -> None:
        for j in range(i + 1, self.variables.size):
            if sorted_domains[j, MIN] >= a:
                new_domains[sorted_vars[j], MIN] = np.max(sorted_domains[j, MIN], b + 1)

    def __str__(self) -> str:
        return f"alldifferent_puget_n3({self.variables})"
