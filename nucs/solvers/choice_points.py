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

from nucs.constants import DOM_UPDATE_EVENTS, DOM_UPDATE_IDX, MAX, MIN, STATS_IDX_SOLVER_BACKTRACK_NB
from nucs.propagators.propagators import update_propagators


@njit(cache=True)
def cp_init(
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    domains_arr: NDArray,
) -> None:
    """
    Inits the choice points.
    :param domains_stk: the stack of shared domains
    :param not_entailed_propagators_stk: the stack of not entailed propagators
    :param dom_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param domains_arr: the shared domains
    :return: a Numpy array
    """
    domains_stk[0] = domains_arr
    not_entailed_propagators_stk[0] = True
    dom_update_stk[:, :] = 0
    stks_top[0] = 0


@njit(cache=True)
def cp_put(domains_stk: NDArray, not_entailed_propagators_stk: NDArray, stks_top: NDArray) -> None:
    """
    Adds a choice point to the stack of choice points.
    :param domains_stk: the stack of shared domains
    :param not_entailed_propagators_stk: the stack of not entailed propagators
    :param stks_top: the index of the top of the stacks as a Numpy array
    """
    cp_top_idx = stks_top[0]
    domains_stk[cp_top_idx + 1, :, :] = domains_stk[cp_top_idx, :, :]
    not_entailed_propagators_stk[cp_top_idx + 1, :] = not_entailed_propagators_stk[cp_top_idx, :]
    stks_top[0] = cp_top_idx + 1


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
) -> bool:
    """
    Backtracks and updates the problem's domains.
    :param statistics: the statistics array
    :param not_entailed_propagators_stk: the stack of not entailed propagators
    :param dom_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :param triggers: a Numpy array of event masks indexed by shared domain indices and propagators
    :return: true iff it is possible to backtrack
    """
    if stks_top[0] == 0:
        return False
    stks_top[0] -= 1
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    update_propagators(
        triggered_propagators,
        not_entailed_propagators_stk[stks_top[0]],
        triggers,
        dom_update_stk[stks_top[0], DOM_UPDATE_EVENTS],
        dom_update_stk[stks_top[0], DOM_UPDATE_IDX],
    )
    return True


@njit(cache=True)
def fix_choice_points(
    domains_stk: NDArray,
    stks_top: NDArray,
    variables_arr: NDArray,
    offsets_arr: NDArray,
    variable_idx: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized in the choice points.
    :param domains_stk: the stack of shared domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param variables_arr: the variables as an array
    :param offsets_arr: the offsets as an array
    :param variable_idx: the index of the variable being optimized
    :param value: the current optimal value for the variable
    :param bound: the bound being optimized
    """
    dom_idx = variables_arr[variable_idx]
    domains_stk[0 : stks_top[0] + 1, dom_idx, bound] = value + (1 if bound == MIN else -1) - offsets_arr[variable_idx]
    while domains_stk[stks_top[0], dom_idx, MIN] > domains_stk[stks_top[0], dom_idx, MAX]:
        if stks_top[0] == 0:
            return False
        stks_top[0] -= 1
    return True


@njit(cache=True)
def fix_top_choice_point(
    domains_stk: NDArray,
    stks_top: NDArray,
    variables_arr: NDArray,
    offsets_arr: NDArray,
    variable_idx: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized in the top choice point.
    :param domains_stk: the stack of shared domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param variables_arr: the variables as an array
    :param offsets_arr: the offsets as an array
    :param variable_idx: the index of the variable being optimized
    :param value: the current optimal value for the variable
    :param bound: the bound being optimized
    """
    top = stks_top[0]
    dom_idx = variables_arr[variable_idx]
    domains_stk[top, dom_idx, bound] = value + (1 if bound == MIN else -1) - offsets_arr[variable_idx]
    return domains_stk[top, dom_idx, MIN] <= domains_stk[top, dom_idx, MAX]
