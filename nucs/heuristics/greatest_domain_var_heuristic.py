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
