import sys
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
        """
        Inits the heuristic.
        :param variable_heuristic: a function for choosing the variable
        :param domain_heuristic: a function for choosing a new domain for this variable
        """
        self.variable_heuristic = variable_heuristic
        self.domain_heuristic = domain_heuristic

    def choose(self, shared_domains: NDArray, domain_indices: NDArray) -> Tuple[NDArray, NDArray]:
        var_idx = self.variable_heuristic(shared_domains, domain_indices)
        return self.domain_heuristic(shared_domains, domain_indices[var_idx])


@jit(nopython=True, cache=True)
def first_not_instantiated_var_heuristic(shared_domains: NDArray, domain_indices: NDArray) -> int:
    """
    Chooses the first non instantiated variable.
    :param shared_domains: the shared domains of the problem
    :param domain_indices: the domain indices of the problem variables
    :return: the index of the variable
    """
    for var_idx, domain_index in enumerate(domain_indices):
        domain = shared_domains[domain_index]
        if domain[MIN] < domain[MAX]:
            return var_idx
    return -1  # cannot happen


@jit(nopython=True, cache=True)
def smallest_domain_var_heuristic(shared_domains: NDArray, domain_indices: NDArray) -> int:
    """
    Chooses the variable with the smallest domain and which is not instantiated.
    :param shared_domains: the shared domains of the problem
    :param domain_indices: the domain indices of the problem variables
    :return: the index of the variable
    """
    min_size = sys.maxsize
    min_idx = -1
    for var_idx, domain_index in enumerate(domain_indices):
        domain = shared_domains[domain_index]
        size = domain[MAX] - domain[MIN]  # actually this is size - 1
        if 0 < size < min_size:
            min_idx = var_idx
            min_size = size
    return min_idx


@jit(nopython=True, cache=True)
def min_value_dom_heuristic(shared_domains: NDArray, domain_idx: int) -> Tuple[NDArray, NDArray]:
    """
    Chooses the first value of the domain
    :param shared_domains: the shared domains of the problem
    :param domain_idx: the index of the domain
    :return: the new shared domain to be added to the choice point and the changes to the actual domains
    """
    domains = shared_domains.copy()
    min_value = shared_domains[domain_idx, MIN]
    domains[domain_idx, MIN] = min_value + 1
    shared_domains[domain_idx, MAX] = min_value
    shr_changes = np.zeros((len(shared_domains), 2), dtype=np.bool)
    shr_changes[domain_idx, MAX] = True
    return domains, shr_changes


@jit(nopython=True, cache=True)
def split_low_dom_heuristic(shared_domains: NDArray, domain_idx: int) -> Tuple[NDArray, NDArray]:
    """
    Chooses the first half of the domain
    :param shared_domains: the shared domains of the problem
    :param domain_idx: the index of the domain
    :return: the new shared domain to be added to the choice point and the changes to the actual domains
    """
    domains = shared_domains.copy()
    mid_value = (shared_domains[domain_idx, MIN] + shared_domains[domain_idx, MAX]) // 2
    domains[domain_idx, MIN] = mid_value + 1
    shared_domains[domain_idx, MAX] = mid_value
    shr_changes = np.zeros((len(shared_domains), 2), dtype=np.bool)
    shr_changes[domain_idx, MAX] = True
    return domains, shr_changes
