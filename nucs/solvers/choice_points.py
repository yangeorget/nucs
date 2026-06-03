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

from nucs.constants import DOM_UPDATE_EVENTS, DOM_UPDATE_VARIABLE, MAX, MIN, STATS_IDX_SOLVER_BACKTRACK_NB
from nucs.propagators.propagators import update_propagators


@njit(cache=True, fastmath=True)
def cp_init(
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    domains_arr: NDArray,
    unbound_variable_nb: int,
) -> None:
    """
    Initializes the choice points.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size,
                             the following cells hold the indices of the entailed propagators in entailment order
    :type entailment_trail: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of unbound variable nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param domains_arr: the domains
    :type domains_arr: NDArray
    :param unbound_variable_nb: the number of unbound variables
    :type unbound_variable_nb: int
    """
    domains_stk[0] = domains_arr
    entailed_propagator_depths.fill(-1)
    entailment_trail[0] = 0
    domain_update_stk.fill(0)
    unbound_variable_nb_stk[0] = unbound_variable_nb
    stks_top[0] = 0


@njit(cache=True, fastmath=True)
def cp_put(
    domains_stk: NDArray, entailed_propagator_depths: NDArray, unbound_variable_nb_stk: NDArray, top: int
) -> None:
    """
    Adds a choice point to the stack of choice points.

    Entailment is monotonic within a branch, so descending to a deeper choice point requires no entailment
    bookkeeping: the depths recorded so far stay valid and entailed_propagator_depths is left untouched.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, unused here
    :type entailed_propagator_depths: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    """
    domains_stk[top + 1] = domains_stk[top]  # copy the domains
    unbound_variable_nb_stk[top + 1] = unbound_variable_nb_stk[top]  # copy the number of unbound variables


@njit(cache=True, fastmath=True)
def unwind_entailment_trail(entailed_propagator_depths: NDArray, entailment_trail: NDArray, top: int) -> None:
    """
    Reactivates the propagators that were entailed below the current top.

    The trail is ordered by non-decreasing entailment depth, so it suffices to pop, from the top of the trail,
    every propagator whose entailment depth is strictly greater than the current top and reset it to active.

    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    """
    size = entailment_trail[0]
    while size > 0 and entailed_propagator_depths[entailment_trail[size]] > top:
        entailed_propagator_depths[entailment_trail[size]] = -1
        size -= 1
    entailment_trail[0] = size


@njit(cache=True, fastmath=True)
def backtrack(
    statistics: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
    complexities: NDArray,
    propagator_nb: int,
) -> bool:
    """
    Backtracks and updates the problem's domains.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :type triggered_propagators: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param complexities: the propagation queue bucket priorities indexed by propagators
    :type complexities: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    if stks_top[0] == 0:
        return False
    stks_top[0] -= 1
    top = stks_top[0]
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    unwind_entailment_trail(entailed_propagator_depths, entailment_trail, top)
    domain_update = domain_update_stk[top]
    update_propagators(
        triggered_propagators,
        entailed_propagator_depths,
        triggers[domain_update[DOM_UPDATE_VARIABLE], domain_update[DOM_UPDATE_EVENTS]],
        complexities,
        propagator_nb,
    )
    return True


@njit(cache=True, fastmath=True)
def fix_choice_points(
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized in the choice points.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param variable: the variable being optimized
    :type variable: int
    :param value: the current optimal value for the variable
    :type value: int
    :param bound: the bound being optimized
    :type bound: int

    :return: true iff at least one choice point remains
    :rtype: bool
    """
    if stks_top[0] == 0:
        return False
    stks_top[0] -= 1
    for stks_idx in range(stks_top[0], -1, -1):
        was_bound = domains_stk[stks_idx, variable, MAX] == domains_stk[stks_idx, variable, MIN]
        if bound == MIN:
            domains_stk[stks_idx, variable, bound] = max(value + 1, domains_stk[stks_idx, variable, bound])
        else:
            domains_stk[stks_idx, variable, bound] = min(value - 1, domains_stk[stks_idx, variable, bound])
        range_sz = domains_stk[stks_idx, variable, MAX] - domains_stk[stks_idx, variable, MIN]
        if range_sz < 0:
            if stks_top[0] == 0:
                unwind_entailment_trail(entailed_propagator_depths, entailment_trail, stks_top[0])
                return False
            stks_top[0] -= 1
        elif range_sz == 0:
            if not was_bound:
                unbound_variable_nb_stk[stks_idx] -= 1
    unwind_entailment_trail(entailed_propagator_depths, entailment_trail, stks_top[0])
    return True


@njit(cache=True, fastmath=True)
def fix_choice_point(
    domains_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    variable: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized in the top choice point.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param variable:  the variable being optimized
    :type variable: int
    :param value: the current optimal value for the variable
    :type value: int
    :param bound: the bound being optimized
    :type bound: int

    :return: true iff the resulting domain is non-empty
    :rtype: bool
    """
    if bound == MIN:
        domains_stk[0, variable, bound] = value + 1
    else:
        domains_stk[0, variable, bound] = value - 1
    range_sz = domains_stk[0, variable, MAX] - domains_stk[0, variable, MIN]
    if range_sz < 0:
        return False
    if range_sz == 0:
        unbound_variable_nb_stk[0] -= 1
    return True
