import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
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


@njit(cache=True)
def last_not_instantiated_var_heuristic(shr_domains: NDArray, dom_indices: NDArray) -> int:
    """
    Chooses the last non instantiated variable.
    :param shr_domains: the shared domains of the problem
    :param dom_indices: the domain indices of the problem variables
    :return: the index of the variable
    """
    for var_idx in range(len(dom_indices) - 1, -1, -1):
        dom_index = dom_indices[var_idx]
        if shr_domains[dom_index, MIN] < shr_domains[dom_index, MAX]:
            return var_idx
    return -1  # cannot happen


@njit(cache=True)
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


@njit(cache=True)
def greatest_domain_var_heuristic(shr_domains: NDArray, dom_indices: NDArray) -> int:
    """
    Chooses the variable with the greatest domain and which is not instantiated.
    :param shr_domains: the shared domains of the problem
    :param dom_indices: the domain indices of the problem variables
    :return: the index of the variable
    """
    max_size = 0
    max_idx = -1
    for var_idx, dom_index in enumerate(dom_indices):
        size = shr_domains[dom_index, MAX] - shr_domains[dom_index, MIN]  # actually this is size - 1
        if max_size < size:
            max_idx = var_idx
            max_size = size
    return max_idx


@njit(cache=True)
def min_value_dom_heuristic(
    shr_domains: NDArray, shr_domain_changes: NDArray, shr_domains_copy: NDArray, domain_idx: int
) -> None:
    """
    Chooses the first value of the domain
    :param shr_domains: the shared domains of the problem
    :param: shr_domain_changes: the changes to the shared domains
    :param: shr_domains_copy: the shared domains to be added to the choice point
    :param domain_idx: the index of the domain
    """
    value = shr_domains[domain_idx, MIN]
    shr_domains_copy[domain_idx, MIN] = value + 1
    shr_domains[domain_idx, MAX] = value
    shr_domain_changes[domain_idx, MAX] = True


@njit(cache=True)
def max_value_dom_heuristic(
    shr_domains: NDArray, shr_domain_changes: NDArray, shr_domains_copy: NDArray, domain_idx: int
) -> None:
    """
    Chooses the last value of the domain
    :param shr_domains: the shared domains of the problem
    :param: shr_domain_changes: the changes to the shared domains
    :param: shr_domains_copy: the shared domains to be added to the choice point
    :param domain_idx: the index of the domain
    """
    value = shr_domains[domain_idx, MAX]
    shr_domains_copy[domain_idx, MAX] = value - 1
    shr_domains[domain_idx, MIN] = value
    shr_domain_changes[domain_idx, MIN] = True


@njit(cache=True)
def split_low_dom_heuristic(
    shr_domains: NDArray, shr_domain_changes: NDArray, shr_domains_copy: NDArray, domain_idx: int
) -> None:
    """
    Chooses the first half of the domain
    :param shr_domains: the shared domains of the problem
    :param: shr_domain_changes: the changes to the shared domains
    :param: shr_domains_copy: the shared domains to be added to the choice point
    :param domain_idx: the index of the domain
    """
    value = (shr_domains[domain_idx, MIN] + shr_domains[domain_idx, MAX]) // 2
    shr_domains_copy[domain_idx, MIN] = value + 1
    shr_domains[domain_idx, MAX] = value
    shr_domain_changes[domain_idx, MAX] = True


(
    VAR_HEURISTIC_FIRST_NON_INSTANTIATED,
    VAR_HEURISTIC_LAST_NON_INSTANTIATED,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    VAR_HEURISTIC_GREATEST_DOMAIN,
) = tuple(range(4))

DOM_HEURISTIC_MIN_VALUE, DOM_HEURISTIC_MAX_VALUE, DOM_HEURISTIC_SPLIT_LOW = tuple(range(3))


VAR_HEURISTIC_FCTS = [
    first_not_instantiated_var_heuristic,
    last_not_instantiated_var_heuristic,
    smallest_domain_var_heuristic,
    greatest_domain_var_heuristic,
]

DOM_HEURISTIC_FCTS = [min_value_dom_heuristic, min_value_dom_heuristic, split_low_dom_heuristic]
