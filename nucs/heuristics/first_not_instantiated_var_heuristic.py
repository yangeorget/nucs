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
def first_not_instantiated_var_heuristic(
    decision_domains: NDArray, domains_stk: NDArray, stks_top: NDArray, params: NDArray
) -> int:
    """
    Chooses the first non-instantiated shared domain.
    :param decision_domains: the indices of a subset of the shared domains
    :param domains_stk: the stack of shared domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param params: a two-dimensional parameters array, unused here
    :return: the index of the shared domain
    """
    top = stks_top[0]
    for dom_idx in decision_domains:
        if domains_stk[top, dom_idx, MIN] < domains_stk[top, dom_idx, MAX]:
            return dom_idx
    return -1  # cannot happen
