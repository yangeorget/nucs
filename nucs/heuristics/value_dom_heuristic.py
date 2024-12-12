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

from nucs.constants import (
    DOM_UPDATE_EVENTS,
    DOM_UPDATE_IDX,
    EVENT_MASK_MAX,
    EVENT_MASK_MAX_GROUND,
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_GROUND,
    EVENT_MASK_MIN_MAX_GROUND,
    MAX,
    MIN,
)
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
    params: NDArray,
) -> int:
    """
    Chooses a value given as a parameter.
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :param value: the value
    :param params: a two-dimensional parameters array
    :return: the events
    """
    top = stacks_top[0]
    if value == shr_domains_stack[top, dom_idx, MIN]:
        return min_value_dom_heuristic(
            shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx, params
        )
    if value == shr_domains_stack[top, dom_idx, MAX]:
        return max_value_dom_heuristic(
            shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx, params
        )
    cp_put(shr_domains_stack, not_entailed_propagators_stack, stacks_top)
    cp_put(shr_domains_stack, not_entailed_propagators_stack, stacks_top)
    shr_domains_stack[top + 2, dom_idx, :] = value
    shr_domains_stack[top + 1, dom_idx, MAX] = value - 1
    shr_domains_stack[top, dom_idx, MIN] = value + 1
    dom_update_stack[top + 1, DOM_UPDATE_IDX] = dom_update_stack[top, DOM_UPDATE_IDX] = dom_idx
    dom_update_stack[top + 1, DOM_UPDATE_EVENTS] = (
        EVENT_MASK_MAX_GROUND
        if shr_domains_stack[top + 1, dom_idx, MIN] == shr_domains_stack[top + 1, dom_idx, MAX]
        else EVENT_MASK_MAX
    )
    dom_update_stack[top, DOM_UPDATE_EVENTS] = (
        EVENT_MASK_MIN_GROUND
        if shr_domains_stack[top, dom_idx, MIN] == shr_domains_stack[top, dom_idx, MAX]
        else EVENT_MASK_MIN
    )
    return EVENT_MASK_MIN_MAX_GROUND
