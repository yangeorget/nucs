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

from nucs.constants import (
    DOM_UPDATE_EVENTS,
    DOM_UPDATE_IDX,
    EVENT_MASK_MAX_GROUND,
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_GROUND,
    MAX,
    MIN,
)
from nucs.solvers.choice_points import cp_put


@njit(cache=True)
def min_value_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
    params: NDArray,
) -> int:
    """
    Chooses the min value of the domain.
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :param params: a two-dimensional parameters array, unused here
    :return: the events
    """
    top = stacks_top[0]
    value = shr_domains_stack[top, dom_idx, MIN]
    cp_put(shr_domains_stack, not_entailed_propagators_stack, stacks_top)
    shr_domains_stack[top + 1, dom_idx, MAX] = value
    shr_domains_stack[top, dom_idx, MIN] = value + 1
    dom_update_stack[top, DOM_UPDATE_IDX] = dom_idx
    dom_update_stack[top, DOM_UPDATE_EVENTS] = (
        EVENT_MASK_MIN_GROUND
        if shr_domains_stack[top, dom_idx, MIN] == shr_domains_stack[top, dom_idx, MAX]
        else EVENT_MASK_MIN
    )
    return EVENT_MASK_MAX_GROUND
