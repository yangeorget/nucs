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


@njit(cache=True)
def cp_init(
    domains_stk: NDArray,
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    domains_arr: NDArray,
    unbound_variable_nb: int,
) -> None:
    """
    Initializes the choice points.
    :param domains_stk: the stack of domains
    :param entailed_propagators_stk: the stack of entailed propagators
    :param domain_update_stk: the stack of domain updates
    :param unbound_variable_nb_stk: the stack of unbound variable nb
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param domains: the domains
    :param unbound_variable_nb: the number of unbound variables
    :return: a Numpy array
    """
    domains_stk[0] = domains_arr
    entailed_propagators_stk[0] = False
    domain_update_stk.fill(0)
    unbound_variable_nb_stk[0] = unbound_variable_nb
    stks_top[0] = 0


@njit(cache=True)
def cp_put(domains_stk: NDArray, entailed_propagators_stk: NDArray, unbound_variable_nb_stk: NDArray, top: int) -> None:
    """
    Adds a choice point to the stack of choice points.
    :param domains_stk: the stack of domains
    :param entailed_propagators_stk: the stack of entailed propagators
    :param top: the index of the top of the stacks
    """
    domains_stk[top + 1] = domains_stk[top]  # copy the domains
    entailed_propagators_stk[top + 1] = entailed_propagators_stk[top]  # copy the entailed propagators
    unbound_variable_nb_stk[top + 1] = unbound_variable_nb_stk[top]  # copy the number of unbound variables


@njit(cache=True)
def backtrack(
    propagator_nb: int,
    statistics: NDArray,
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
) -> bool:
    """
    Backtracks and updates the problem's domains.
    :param statistics: the statistics array
    :param entailed_propagators_stk: the stack of entailed propagators
    :param domain_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :return: true iff it is possible to backtrack
    """
    if stks_top[0] == 0:
        return False
    stks_top[0] -= 1
    top = stks_top[0]
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    domain_update = domain_update_stk[top]
    update_propagators(
        propagator_nb,
        triggered_propagators,
        entailed_propagators_stk[top],
        triggers[domain_update[DOM_UPDATE_VARIABLE], domain_update[DOM_UPDATE_EVENTS]],
    )
    return True


@njit(cache=True)
def fix_choice_points(
    domains_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized in the choice points.
    :param domains_stk: the stack of domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param variable: the variable being optimized
    :param value: the current optimal value for the variable
    :param bound: the bound being optimized
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
                return False
            stks_top[0] -= 1
        elif range_sz == 0:
            if not was_bound:
                unbound_variable_nb_stk[stks_idx] -= 1
    return True


@njit(cache=True)
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
    :param variable:  the variable being optimized
    :param value: the current optimal value for the variable
    :param bound: the bound being optimized
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
