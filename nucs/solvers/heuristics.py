###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024 - Yan Georget
###############################################################################
import sys
from typing import Callable

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def first_not_instantiated_var_heuristic(shr_domains_stack: NDArray, stacks_top: NDArray) -> int:
    """
    Chooses the first non-instantiated shared domain.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    return first_not_instantiated_var_heuristic_from_index(shr_domains_stack, stacks_top, 0)


@njit(cache=True)
def first_not_instantiated_var_heuristic_from_index(
    shr_domains_stack: NDArray, stacks_top: NDArray, start_idx: int
) -> int:
    """
    Chooses the first non-instantiated shared domain.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param start_idx: the index where to start the search
    :return: the index of the shared domain
    """
    cp_top_idx = stacks_top[0]
    for dom_idx in range(start_idx, len(shr_domains_stack[0])):
        if shr_domains_stack[cp_top_idx, dom_idx, MIN] < shr_domains_stack[cp_top_idx, dom_idx, MAX]:
            return dom_idx
    return -1  # cannot happen


@njit(cache=True)
def last_not_instantiated_var_heuristic(shr_domains_stack: NDArray, stacks_top: NDArray) -> int:
    """
    Chooses the last non-instantiated shared domain.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    cp_top_idx = stacks_top[0]
    for dom_idx in range(len(shr_domains_stack[0]) - 1, -1, -1):
        if shr_domains_stack[cp_top_idx, dom_idx, MIN] < shr_domains_stack[cp_top_idx, dom_idx, MAX]:
            return dom_idx
    return -1  # cannot happen


@njit(cache=True)
def smallest_domain_var_heuristic(shr_domains_stack: NDArray, stacks_top: NDArray) -> int:
    """
    Chooses the smallest shared domain and which is not instantiated.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    min_size = sys.maxsize
    min_idx = -1
    for dom_idx, shr_domain in enumerate(shr_domains_stack[stacks_top[0]]):
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if 0 < size < min_size:
            min_idx = dom_idx
            min_size = size
    return min_idx


@njit(cache=True)
def greatest_domain_var_heuristic(shr_domains_stack: NDArray, stacks_top: NDArray) -> int:
    """
    Chooses the greatest shared domain and which is not instantiated.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    max_size = 0
    max_idx = -1
    for dom_index, shr_domain in enumerate(shr_domains_stack[stacks_top[0]]):
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if max_size < size:
            max_idx = dom_index
            max_size = size
    return max_idx


@njit(cache=True)
def min_value_dom_heuristic(
    shr_domains_stack: NDArray, dom_update_stack: NDArray, stacks_top: NDArray, dom_idx: int
) -> int:
    """
    Chooses the first value of the domain.
    :param shr_domains_stack: the stack of shared domains
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :return: the bound which is modified
    """
    cp_top_idx = stacks_top[0]
    value = shr_domains_stack[cp_top_idx, dom_idx, MIN]
    shr_domains_stack[cp_top_idx - 1, dom_idx, MIN] = value + 1
    shr_domains_stack[cp_top_idx, dom_idx, MAX] = value
    dom_update_stack[cp_top_idx - 1, 1] = MAX
    return MAX


@njit(cache=True)
def max_value_dom_heuristic(
    shr_domains_stack: NDArray, dom_update_stack: NDArray, stacks_top: NDArray, dom_idx: int
) -> int:
    """
    Chooses the last value of the domain.
    :param shr_domains_stack: the stack of shared domains
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :return: the bound which is modified
    """
    cp_top_idx = stacks_top[0]
    value = shr_domains_stack[cp_top_idx, dom_idx, MAX]
    shr_domains_stack[cp_top_idx - 1, dom_idx, MAX] = value - 1
    shr_domains_stack[cp_top_idx, dom_idx, MIN] = value
    dom_update_stack[cp_top_idx - 1, 1] = MIN
    return MIN


@njit(cache=True)
def split_low_dom_heuristic(
    shr_domains_stack: NDArray, dom_update_stack: NDArray, stacks_top: NDArray, dom_idx: int
) -> int:
    """
    Chooses the first half of the domain.
    :param shr_domains_stack: the stack of shared domains
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :return: the bound which is modified
    """
    cp_top_idx = stacks_top[0]
    value = (shr_domains_stack[cp_top_idx, dom_idx, MIN] + shr_domains_stack[cp_top_idx, dom_idx, MAX]) // 2
    shr_domains_stack[cp_top_idx - 1, dom_idx, MIN] = value + 1
    shr_domains_stack[cp_top_idx, dom_idx, MAX] = value
    dom_update_stack[cp_top_idx - 1, 1] = MAX
    return MAX


VAR_HEURISTIC_FCTS = []
DOM_HEURISTIC_FCTS = []


def register_var_heuristic(var_heuristic_fct: Callable) -> int:
    """
    Register a variable heuristic by adding it function to the corresponding list of functions.
    :param var_heuristic_fct: a function that implements the variable heuristic
    :return: the index of the variable heuristic
    """
    VAR_HEURISTIC_FCTS.append(var_heuristic_fct)
    return len(VAR_HEURISTIC_FCTS) - 1


def register_dom_heuristic(dom_heuristic_fct: Callable) -> int:
    """
    Register a domain heuristic by adding it function to the corresponding list of functions.
    :param dom_heuristic_fct: a function that implements the domain heuristic
    :return: the index of the domain heuristic
    """
    DOM_HEURISTIC_FCTS.append(dom_heuristic_fct)
    return len(DOM_HEURISTIC_FCTS) - 1


VAR_HEURISTIC_FIRST_NOT_INSTANTIATED = register_var_heuristic(first_not_instantiated_var_heuristic)
VAR_HEURISTIC_LAST_NOT_INSTANTIATED = register_var_heuristic(last_not_instantiated_var_heuristic)
VAR_HEURISTIC_SMALLEST_DOMAIN = register_var_heuristic(smallest_domain_var_heuristic)
VAR_HEURISTIC_GREATEST_DOMAIN = register_var_heuristic(greatest_domain_var_heuristic)

DOM_HEURISTIC_MIN_VALUE = register_dom_heuristic(min_value_dom_heuristic)
DOM_HEURISTIC_MAX_VALUE = register_dom_heuristic(max_value_dom_heuristic)
DOM_HEURISTIC_SPLIT_LOW = register_dom_heuristic(split_low_dom_heuristic)
