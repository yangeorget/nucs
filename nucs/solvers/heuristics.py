import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def first_not_instantiated_var_heuristic(shr_domains: NDArray) -> int:
    """
    Chooses the first non-instantiated shared domain.
    :param shr_domains: the shared domains of the problem
    :return: the index of the shared domain
    """
    for dom_idx, shr_domain in enumerate(shr_domains):
        if shr_domain[MIN] < shr_domain[MAX]:
            return dom_idx
    return -1  # cannot happen


@njit(cache=True)
def last_not_instantiated_var_heuristic(shr_domains: NDArray) -> int:
    """
    Chooses the last non-instantiated shared domain.
    :param shr_domains: the shared domains of the problem
    :return: the index of the shared domain
    """
    for dom_idx in range(len(shr_domains) - 1, -1, -1):
        shr_domain = shr_domains[dom_idx]
        if shr_domain[MIN] < shr_domain[MAX]:
            return dom_idx
    return -1  # cannot happen


@njit(cache=True)
def smallest_domain_var_heuristic(shr_domains: NDArray) -> int:
    """
    Chooses the smallest shared domain and which is not instantiated.
    :param shr_domains: the shared domains of the problem
    :return: the index of the shared domain
    """
    min_size = sys.maxsize
    min_idx = -1
    for dom_idx, shr_domain in enumerate(shr_domains):
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if 0 < size < min_size:
            min_idx = dom_idx
            min_size = size
    return min_idx


@njit(cache=True)
def greatest_domain_var_heuristic(shr_domains: NDArray) -> int:
    """
    Chooses the greatest shared domain and which is not instantiated.
    :param shr_domains: the shared domains of the problem
    :return: the index of the shared domain
    """
    max_size = 0
    max_idx = -1
    for dom_index, shr_domain in enumerate(shr_domains):
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if max_size < size:
            max_idx = dom_index
            max_size = size
    return max_idx


@njit(cache=True)
def min_value_dom_heuristic(shr_domain: NDArray, shr_domain_copy: NDArray) -> int:
    """
    Chooses the first value of the domain.
    """
    value = shr_domain[MIN]
    shr_domain_copy[MIN] = value + 1
    shr_domain[MAX] = value
    return MAX


@njit(cache=True)
def max_value_dom_heuristic(shr_domain: NDArray, shr_domain_copy: NDArray) -> int:
    """
    Chooses the last value of the domain.
    """
    value = shr_domain[MAX]
    shr_domain_copy[MAX] = value - 1
    shr_domain[MIN] = value
    return MIN


@njit(cache=True)
def split_low_dom_heuristic(shr_domain: NDArray, shr_domain_copy: NDArray) -> int:
    """
    Chooses the first half of the domain.
    """
    value = (shr_domain[MIN] + shr_domain[MAX]) // 2
    shr_domain_copy[MIN] = value + 1
    shr_domain[MAX] = value
    return MAX
