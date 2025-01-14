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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def greatest_domain_var_heuristic(
    decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray, params: NDArray
) -> int:
    """
    Chooses the greatest shared domain and which is not instantiated.
    :param decision_domains: the indices of a subset of the shared domains
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param params: a two-dimensional parameters array, unused here
    :return: the index of the shared domain
    """
    best_score = 0
    best_idx = -1
    top = stacks_top[0]
    for dom_idx in decision_domains:
        shr_domain = shr_domains_stack[top, dom_idx]
        score = shr_domain[MAX] - shr_domain[MIN]  # this is size - 1
        if best_score < score:
            best_idx = dom_idx
            best_score = score
    return best_idx
