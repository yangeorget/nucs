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
from nucs.heuristics.value_dom_heuristic import value_dom_heuristic


@njit(cache=True)
def mid_value_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
    params: NDArray,
) -> int:
    """
    Chooses the middle value of the domain.
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :param params: a two-dimensional parameters array, unused here
    :return: the events
    """
    return value_dom_heuristic(
        shr_domains_stack,
        not_entailed_propagators_stack,
        dom_update_stack,
        stacks_top,
        dom_idx,
        (shr_domains_stack[stacks_top[0], dom_idx, MIN] + shr_domains_stack[stacks_top[0], dom_idx, MAX]) // 2,
        params,
    )
