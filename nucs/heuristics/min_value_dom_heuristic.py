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

from nucs.constants import BOUNDS, DOM_IDX, MAX, MIN
from nucs.solvers.choice_points import cp_put


@njit(cache=True)
def min_value_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
) -> int:
    """
    Chooses the min value of the domain.
    :param shr_domains_stack: the stack of shared domains
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :return: the bounds which are modified
    """
    cp_cur_idx = stacks_top[0]
    value = shr_domains_stack[cp_cur_idx, dom_idx, MIN]
    cp_put(shr_domains_stack, not_entailed_propagators_stack, stacks_top)
    shr_domains_stack[cp_cur_idx + 1, dom_idx, MAX] = value
    shr_domains_stack[cp_cur_idx, dom_idx, MIN] = value + 1
    dom_update_stack[cp_cur_idx, DOM_IDX] = dom_idx
    dom_update_stack[cp_cur_idx, BOUNDS] = MIN
    return MAX
