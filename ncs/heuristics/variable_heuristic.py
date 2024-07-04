from typing import Callable, Tuple

import numpy as np
from numba import jit
from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.utils import MAX, MIN


class VariableHeuristic(Heuristic):
    """
    Chooses a variable and one or several values for this variable.
    """

    def __init__(self, variable_heuristic: Callable, domain_heuristic: Callable):
        self.variable_heuristic = variable_heuristic
        self.domain_heuristic = domain_heuristic

    def choose(self, shared_domains: NDArray, domain_indices: NDArray) -> Tuple[NDArray, NDArray]:
        var_idx = self.variable_heuristic(shared_domains, domain_indices)
        return self.domain_heuristic(shared_domains, domain_indices[var_idx])


@jit(nopython=True, cache=True)
def first_not_instantiated_variable_heuristic(shared_domains: NDArray, domain_indices: NDArray) -> int:
    for var_idx, domain_index in enumerate(domain_indices):
        domain = shared_domains[domain_index]
        if domain[MIN] < domain[MAX]:
            return var_idx
    return -1  # cannot happen


def smallest_domain_variable_heuristic(shared_domains: NDArray, domain_indices: NDArray) -> int:
    return np.ma.argmin(
        np.ma.masked_equal(shared_domains[domain_indices, MAX] - shared_domains[domain_indices, MIN], 0, copy=False)
    )


@jit(nopython=True, cache=True)
def min_value_domain_heuristic(shared_domains: NDArray, domain_idx: int) -> Tuple[NDArray, NDArray]:
    domains = np.copy(shared_domains)
    min_value = shared_domains[domain_idx, MIN]
    domains[domain_idx, MIN] = min_value + 1
    shared_domains[domain_idx, MAX] = min_value
    shr_changes = np.zeros((len(shared_domains), 2), dtype=np.bool)
    shr_changes[domain_idx, MAX] = True
    return domains, shr_changes
