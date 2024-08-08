import sys
from typing import Callable, Tuple

import numpy as np
from numba import jit  # type: ignore
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

    def choose(self, shr_domains: NDArray, dom_indices: NDArray) -> Tuple[NDArray, NDArray]:
        var_idx = self.variable_heuristic(shr_domains, dom_indices)
        return self.domain_heuristic(shr_domains, dom_indices[var_idx])


@jit(nopython=True, cache=True)
def first_not_instantiated_var_heuristic(shr_domains: NDArray, dom_indices: NDArray) -> int:
    """
    Chooses the first non instantiated variable.
    :param shr_domains: the shared domains of the problem
    :param dom_indices: the domain indices of the problem variables
    :return: the index of the variable
    """
    for var_idx, dom_index in enumerate(dom_indices):
        if shr_domains[dom_index, MIN] < shr_domains[dom_index, MAX]:
            return var_idx
    return -1  # cannot happen


@jit(nopython=True, cache=True)
def smallest_domain_var_heuristic(shr_domains: NDArray, dom_indices: NDArray) -> int:
    """
    Chooses the variable with the smallest domain and which is not instantiated.
    :param shr_domains: the shared domains of the problem
    :param dom_indices: the domain indices of the problem variables
    :return: the index of the variable
    """
    min_size = sys.maxsize
    min_idx = -1
    for var_idx, dom_index in enumerate(dom_indices):
        size = shr_domains[dom_index, MAX] - shr_domains[dom_index, MIN]  # actually this is size - 1
        if 0 < size < min_size:
            min_idx = var_idx
            min_size = size
    return min_idx


@jit(nopython=True, cache=True)
def min_value_dom_heuristic(shr_domains: NDArray, domain_idx: int) -> Tuple[NDArray, NDArray]:
    """
    Chooses the first value of the domain
    :param shr_domains: the shared domains of the problem
    :param domain_idx: the index of the domain
    :return: the new shared domain to be added to the choice point and the changes to the actual domains
    """
    shr_domains_copy = shr_domains.copy()
    min_value = shr_domains[domain_idx, MIN]
    shr_domains_copy[domain_idx, MIN] = min_value + 1
    shr_domains[domain_idx, MAX] = min_value
    shr_domain_changes = np.zeros((len(shr_domains), 2), dtype=np.bool)
    shr_domain_changes[domain_idx, MAX] = True
    return shr_domains_copy, shr_domain_changes


@jit(nopython=True, cache=True)
def split_low_dom_heuristic(shr_domains: NDArray, domain_idx: int) -> Tuple[NDArray, NDArray]:
    """
    Chooses the first half of the domain
    :param shr_domains: the shared domains of the problem
    :param domain_idx: the index of the domain
    :return: the new shared domain to be added to the choice point and the changes to the actual domains
    """
    shr_domains_copy = shr_domains.copy()
    mid_value = (shr_domains[domain_idx, MIN] + shr_domains[domain_idx, MAX]) // 2
    shr_domains_copy[domain_idx, MIN] = mid_value + 1
    shr_domains[domain_idx, MAX] = mid_value
    shr_domain_changes = np.zeros((len(shr_domains), 2), dtype=np.bool)
    shr_domain_changes[domain_idx, MAX] = True
    return shr_domains_copy, shr_domain_changes
