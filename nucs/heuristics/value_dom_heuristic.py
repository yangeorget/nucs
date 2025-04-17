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
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    dom_idx: int,
    value: int,
    params: NDArray,
) -> int:
    """
    Chooses a value given as a parameter.
    :param domains_stk: the stack of shared domains
    :param not_entailed_propagators_stk: the stack of not entailed propagators
    :param dom_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :param value: the value
    :param params: a two-dimensional parameter array
    :return: the events
    """
    top = stks_top[0]
    if value == domains_stk[top, dom_idx, MIN]:
        return min_value_dom_heuristic(
            domains_stk, not_entailed_propagators_stk, dom_update_stk, stks_top, dom_idx, params
        )
    if value == domains_stk[top, dom_idx, MAX]:
        return max_value_dom_heuristic(
            domains_stk, not_entailed_propagators_stk, dom_update_stk, stks_top, dom_idx, params
        )
    cp_put(domains_stk, not_entailed_propagators_stk, stks_top)
    cp_put(domains_stk, not_entailed_propagators_stk, stks_top)
    domains_stk[top + 2, dom_idx, :] = value
    domains_stk[top + 1, dom_idx, MAX] = value - 1
    domains_stk[top, dom_idx, MIN] = value + 1
    dom_update_stk[top + 1, DOM_UPDATE_IDX] = dom_update_stk[top, DOM_UPDATE_IDX] = dom_idx
    dom_update_stk[top + 1, DOM_UPDATE_EVENTS] = (
        EVENT_MASK_MAX_GROUND
        if domains_stk[top + 1, dom_idx, MIN] == domains_stk[top + 1, dom_idx, MAX]
        else EVENT_MASK_MAX
    )
    dom_update_stk[top, DOM_UPDATE_EVENTS] = (
        EVENT_MASK_MIN_GROUND if domains_stk[top, dom_idx, MIN] == domains_stk[top, dom_idx, MAX] else EVENT_MASK_MIN
    )
    return EVENT_MASK_MIN_MAX_GROUND
