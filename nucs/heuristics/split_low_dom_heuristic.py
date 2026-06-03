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
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_GROUND,
    MAX,
    MIN,
)
from nucs.solvers.choice_points import cp_put


@njit(cache=True, fastmath=True)
def split_low_dom_heuristic(
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    params: NDArray,
) -> int:
    """
    Chooses the first half of the domain.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the events
    :rtype: int
    """
    top = stks_top[0]
    cp_put(domains_stk, entailed_propagator_depths, unbound_variable_nb_stk, top)
    value = (domains_stk[top, variable, MIN] + domains_stk[top, variable, MAX]) >> 1
    domains_stk[top + 1, variable, MAX] = value
    domains_stk[top, variable, MIN] = value + 1
    if domains_stk[top + 1, variable, MIN] == domains_stk[top + 1, variable, MAX]:
        unbound_variable_nb_stk[top + 1] -= 1
    domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
    if domains_stk[top, variable, MIN] == domains_stk[top, variable, MAX]:
        domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MIN_GROUND
        unbound_variable_nb_stk[top] -= 1
    else:
        domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MIN
    stks_top[0] = top + 1
    return EVENT_MASK_MAX
