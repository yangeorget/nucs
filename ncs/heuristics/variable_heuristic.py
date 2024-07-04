from typing import Callable, List, Optional, Tuple

import numpy as np
from numba import jit
from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.problems.problem import Problem
from ncs.utils import MAX, MIN


class VariableHeuristic(Heuristic):
    """
    Chooses a variable and one or several values for this variable.
    """

    def __init__(self, variable_heuristic: Callable, domain_heuristic: Callable):
        self.variable_heuristic = variable_heuristic
        self.domain_heuristic = domain_heuristic

    def choose(self, choice_points: List[NDArray], problem: Problem) -> Optional[NDArray]:
        var_idx = self.variable_heuristic(problem.variable_nb, problem.shared_domains, problem.domain_indices)
        if var_idx == -1:
            return None
        domains, changes = self.domain_heuristic(problem.shared_domains, problem.domain_indices[var_idx])
        choice_points.append(domains)
        return changes


@jit(nopython=True, nogil=True, cache=True)
def first_not_instantiated_variable_heuristic(
    variable_nb: int, shared_domains: NDArray, domain_indices: NDArray
) -> int:
    for var_idx in range(variable_nb):
        domain = shared_domains[domain_indices[var_idx]]
        if domain[MIN] < domain[MAX]:
            return var_idx
    return -1


def smallest_domain_variable_heuristic(variable_nb: int, shared_domains: NDArray, domain_indices: NDArray) -> int:
    valid_domain_sizes = np.ma.masked_equal(
        shared_domains[domain_indices, MAX] - shared_domains[domain_indices, MIN], 0, copy=False
    )
    if valid_domain_sizes.count() == 0:
        return -1
    return np.ma.argmin(valid_domain_sizes)

@jit(nopython=True, nogil=True, cache=True)
def min_value_domain_heuristic(shared_domains: NDArray, domain_idx: int) -> Tuple[NDArray, NDArray]:
    domains = np.copy(shared_domains)
    domains[domain_idx, MIN] += 1
    shared_domains[domain_idx, MAX] = shared_domains[domain_idx, MIN]
    shr_changes = np.zeros((len(shared_domains), 2), dtype=np.bool)
    shr_changes[domain_idx, MAX] = True
    return domains, shr_changes
