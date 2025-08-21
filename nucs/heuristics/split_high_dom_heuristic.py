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
    DOM_UPDATE_VARIABLE,
    EVENT_MASK_MAX,
    EVENT_MASK_MAX_GROUND,
    EVENT_MASK_MIN,
    MAX,
    MIN,
)
from nucs.solvers.choice_points import cp_put


@njit(cache=True)
def split_high_dom_heuristic(
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stacks_top: NDArray,
    variable: int,
    params: NDArray,
) -> int:
    """
    Chooses the second half of the domain.
    :param domains_stk: the stack of domains
    :param not_entailed_propagators_stk: the stack of not entailed propagators
    :param dom_update_stk: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param variable: the variable
    :param params: a two-dimensional parameters array, unused here
    :return: the events
    """
    top = stacks_top[0]
    value = (domains_stk[top, variable, MIN] + domains_stk[top, variable, MAX]) // 2
    cp_put(domains_stk, not_entailed_propagators_stk, stacks_top)
    domains_stk[top + 1, variable, MIN] = value + 1
    domains_stk[top, variable, MAX] = value
    dom_update_stk[top, DOM_UPDATE_VARIABLE] = variable
    dom_update_stk[top, DOM_UPDATE_EVENTS] = (
        EVENT_MASK_MAX_GROUND if domains_stk[top, variable, MIN] == domains_stk[top, variable, MAX] else EVENT_MASK_MAX
    )
    return EVENT_MASK_MIN
