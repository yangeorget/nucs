####################################################################
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

from nucs.constants import BOUNDS, DOM_IDX, MAX, MIN, MIN_MAX
from nucs.heuristics.max_value_dom_heuristic import max_value_dom_heuristic
from nucs.heuristics.min_value_dom_heuristic import min_value_dom_heuristic
from nucs.solvers.choice_points import cp_put


@njit(cache=True)
def value_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
    value: int,
) -> int:
    """
    :param shr_domains_stack: the stack of shared domains
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :return: the bound which is modified
    """
    cp_cur_idx = stacks_top[0]
    if value == shr_domains_stack[cp_cur_idx, dom_idx, MIN]:
        return min_value_dom_heuristic(
            shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx
        )
    if value == shr_domains_stack[cp_cur_idx, dom_idx, MAX]:
        return max_value_dom_heuristic(
            shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx
        )
    cp_put(shr_domains_stack, not_entailed_propagators_stack, stacks_top)
    cp_put(shr_domains_stack, not_entailed_propagators_stack, stacks_top)
    shr_domains_stack[cp_cur_idx + 2, dom_idx, :] = value
    shr_domains_stack[cp_cur_idx + 1, dom_idx, MAX] = value - 1
    shr_domains_stack[cp_cur_idx, dom_idx, MIN] = value + 1
    dom_update_stack[cp_cur_idx + 1, DOM_IDX] = dom_idx
    dom_update_stack[cp_cur_idx + 1, BOUNDS] = MIN_MAX
    dom_update_stack[cp_cur_idx, DOM_IDX] = dom_idx
    dom_update_stack[cp_cur_idx, BOUNDS] = MIN_MAX
    return MIN_MAX