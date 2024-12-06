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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def greatest_domain_var_heuristic(
    params: NDArray, decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray
) -> int:
    """
    Chooses the greatest shared domain and which is not instantiated.
    :param params: a two-dimensional parameters array, unused here
    :param decision_domains: the indices of a subset of the shared domains
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    max_size = 0
    max_idx = -1
    cp_top_idx = stacks_top[0]
    for dom_idx in decision_domains:
        shr_domain = shr_domains_stack[cp_top_idx, dom_idx]
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if max_size < size:
            max_idx = dom_idx
            max_size = size
    return max_idx
