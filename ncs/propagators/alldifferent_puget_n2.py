from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.alldifferent_puget_n3 import AlldifferentPugetN3


class AlldifferentPugetN2(AlldifferentPugetN3):
    """
    JF Puget's bound consistency algorithm for the alldifferent constraint with quadratic complexity.
    """

    def insert(self, new_doms: NDArray, i: int, u: NDArray, sorted_doms: NDArray, sorted_vars: NDArray) -> bool:
        u[i] = sorted_doms[i, MIN]
        best_min = self.size + 1
        for j in range(0, i):
            if sorted_doms[j, MIN] < sorted_doms[i, MIN]:
                u[j] += 1
                if u[j] > sorted_doms[i, MAX]:
                    return False
                if u[j] == sorted_doms[i, MAX] and sorted_doms[j, MIN] < best_min:
                    best_min = sorted_doms[j, MIN]
            else:
                u[i] += 1
        if u[i] > sorted_doms[i, MAX]:
            return False
        if u[i] == sorted_doms[i, MAX] and sorted_doms[i, MIN] < best_min:
            best_min = sorted_doms[i, MIN]
        if best_min <= self.size:
            self.increment_min(new_doms, i, best_min, sorted_doms[i, MAX], sorted_doms, sorted_vars)
        return True

    def __str__(self) -> str:
        return f"alldifferent_puget_n2({self.variables})"
