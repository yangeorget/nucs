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
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    value: int,
    params: NDArray,
) -> int:
    """
    Chooses a value given as a parameter.
    :param domains_stk: the stack of domains
    :param entailed_propagators_stk: the stack of entailed propagators
    :param domain_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param variable: the variable
    :param value: the value
    :param params: a two-dimensional parameter array
    :return: the events
    """
    top = stks_top[0]
    if value == domains_stk[top, variable, MIN]:
        return min_value_dom_heuristic(
            domains_stk,
            entailed_propagators_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            params,
        )
    if value == domains_stk[top, variable, MAX]:
        return max_value_dom_heuristic(
            domains_stk,
            entailed_propagators_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            params,
        )
    cp_put(domains_stk, entailed_propagators_stk, unbound_variable_nb_stk, top)
    cp_put(domains_stk, entailed_propagators_stk, unbound_variable_nb_stk, top + 1)
    domains_stk[top + 2, variable] = value
    domains_stk[top + 1, variable, MAX] = value - 1
    domains_stk[top, variable, MIN] = value + 1
    unbound_variable_nb_stk[top + 2] -= 1
    domain_update_stk[top + 1, DOM_UPDATE_VARIABLE] = domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
    if domains_stk[top + 1, variable, MIN] == domains_stk[top + 1, variable, MAX]:
        domain_update_stk[top + 1, DOM_UPDATE_EVENTS] = EVENT_MASK_MAX_GROUND
        unbound_variable_nb_stk[top + 1] -= 1
    else:
        domain_update_stk[top + 1, DOM_UPDATE_EVENTS] = EVENT_MASK_MAX
    if domains_stk[top, variable, MIN] == domains_stk[top, variable, MAX]:
        domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MIN_GROUND
        unbound_variable_nb_stk[top] -= 1
    else:
        domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MIN
    stks_top[0] = top + 2
    return EVENT_MASK_MIN_MAX_GROUND
