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

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def smallest_domain_var_heuristic(decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray) -> int:
    """
    Chooses the smallest shared domain and which is not instantiated.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    min_size = sys.maxsize
    min_idx = -1
    cp_top_idx = stacks_top[0]
    for dom_idx in decision_domains:
        shr_domain = shr_domains_stack[cp_top_idx, dom_idx]
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if 0 < size < min_size:
            min_idx = dom_idx
            min_size = size
    return min_idx
