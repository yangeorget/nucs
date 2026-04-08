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
# Copyright 2024-2026 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOM_UPDATE_EVENTS,
    DOM_UPDATE_VARIABLE,
    EVENT_MASK_MAX,
    EVENT_MASK_MAX_GROUND,
    EVENT_MASK_MIN_GROUND,
    MAX,
    MIN,
)
from nucs.solvers.choice_points import cp_put


@njit(cache=True)
def max_value_dom_heuristic(
    domains_stk: NDArray,
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    params: NDArray,
) -> int:
    """
    Chooses the max value of the domain.
    :param domains_stk: the stack of domains
    :param entailed_propagators_stk: the stack of entailed propagators
    :param domain_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param variable: the variable
    :param params: a two-dimensional parameter array, unused here
    :return: the events
    """
    top = stks_top[0]
    cp_put(domains_stk, entailed_propagators_stk, unbound_variable_nb_stk, top)
    value = domains_stk[top, variable, MAX]
    domains_stk[top + 1, variable, MIN] = value
    domains_stk[top, variable, MAX] = value - 1
    unbound_variable_nb_stk[top + 1] -= 1
    domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
    if domains_stk[top, variable, MIN] == domains_stk[top, variable, MAX]:
        domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MAX_GROUND
        unbound_variable_nb_stk[top] -= 1
    else:
        domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MAX
    stks_top[0] = top + 1
    return EVENT_MASK_MIN_GROUND
