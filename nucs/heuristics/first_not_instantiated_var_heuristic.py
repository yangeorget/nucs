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
def first_not_instantiated_var_heuristic(
    decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray
) -> int:
    """
    Chooses the first non-instantiated shared domain.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    cp_top_idx = stacks_top[0]
    for dom_idx in decision_domains:
        if shr_domains_stack[cp_top_idx, dom_idx, MIN] < shr_domains_stack[cp_top_idx, dom_idx, MAX]:
            return dom_idx
    return -1  # cannot happen
