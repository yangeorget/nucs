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
    EVENT_MASK_GROUND,
    MAX,
    MIN,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
    STATS_IDX_ALG_BC_WITH_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_CHANGE_NB,
    STATS_IDX_ALG_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_NO_CHANGE_NB,
)
from nucs.heuristics.first_not_instantiated_var_heuristic import first_not_instantiated_var_heuristic
from nucs.heuristics.heuristics import max_value_dom_heuristic, min_value_dom_heuristic
from nucs.propagators.propagators import update_propagators
from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm
from nucs.solvers.choice_points import backtrack


@njit(cache=True)
def shave_bound(
    propagator_nb: int,
    bound: int,
    variable: int,
    statistics: NDArray,
    algorithms: NDArray,
    bounds: NDArray,
    props_variables: NDArray,
    props_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    decision_variables: NDArray,
) -> bool:
    events = (
        max_value_dom_heuristic(domains_stk, not_entailed_propagators_stk, dom_update_stk, stks_top, variable, None)
        if bound == MAX
        else min_value_dom_heuristic(
            domains_stk, not_entailed_propagators_stk, dom_update_stk, stks_top, variable, None
        )
    )
    if domains_stk[stks_top[0], variable, MIN] == domains_stk[stks_top[0], variable, MAX]:
        events |= EVENT_MASK_GROUND
    update_propagators(
        propagator_nb, triggered_propagators, not_entailed_propagators_stk[stks_top[0]], triggers, events, variable
    )
    if (
        bound_consistency_algorithm(
            propagator_nb,
            statistics,
            algorithms,
            bounds,
            props_variables,
            props_parameters,
            triggers,
            domains_stk,
            not_entailed_propagators_stk,
            dom_update_stk,
            stks_top,
            triggered_propagators,
            compute_domains_addrs,
            decision_variables,
        )
        == PROBLEM_INCONSISTENT
    ):
        has_shaved = True
    else:
        domains_stk[stks_top[0] - 1, variable, bound] += 1 if bound == MAX else -1
        has_shaved = False
    backtrack(
        propagator_nb,
        statistics,
        not_entailed_propagators_stk,
        dom_update_stk,
        stks_top,
        triggered_propagators,
        triggers,
    )
    return has_shaved


@njit(cache=True)
def shaving_consistency_algorithm(
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    bounds: NDArray,
    props_variables: NDArray,
    props_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    decision_variables: NDArray,
) -> int:
    """
    Shaving consistency algorithm.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param bounds: the bounds indexed by propagators
    :param props_variables: the variables by propagators
    :param props_parameters: the parameters by propagators
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :param domains_stk: a stack of  domains;
    the first level correspond to the current domains, the rest correspond to the choice points
    :param not_entailed_propagators_stk: a stack not entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param dom_update_stk: the stack of domain updates
    :param stks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :param decision_variables: the variables on which decisions will be made
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    statistics[STATS_IDX_ALG_BC_WITH_SHAVING_NB] += 1
    shr_domains_nb = len(domains_stk[0])
    bound = MIN
    has_shaved = True
    start_idx = 0
    while start_idx < shr_domains_nb:
        if has_shaved:
            status = bound_consistency_algorithm(
                propagator_nb,
                statistics,
                algorithms,
                bounds,
                props_variables,
                props_parameters,
                triggers,
                domains_stk,
                not_entailed_propagators_stk,
                dom_update_stk,
                stks_top,
                triggered_propagators,
                compute_domains_addrs,
                decision_variables,
            )
            if status != PROBLEM_UNBOUND:
                return status
        variable = first_not_instantiated_var_heuristic(
            decision_variables[decision_variables >= start_idx], domains_stk, stks_top, None
        )
        if variable == -1:  # all variables after start_idx are instantiated
            break
        statistics[STATS_IDX_ALG_SHAVING_NB] += 1
        has_shaved = shave_bound(
            propagator_nb,
            bound,
            variable,
            statistics,
            algorithms,
            bounds,
            props_variables,
            props_parameters,
            triggers,
            domains_stk,
            not_entailed_propagators_stk,
            dom_update_stk,
            stks_top,
            triggered_propagators,
            compute_domains_addrs,
            decision_variables,
        )
        start_idx = variable
        if has_shaved:
            statistics[STATS_IDX_ALG_SHAVING_CHANGE_NB] += 1
        else:
            statistics[STATS_IDX_ALG_SHAVING_NO_CHANGE_NB] += 1
            # this is one of the many shaving strategies
            bound += 1
            if bound > MAX:
                bound = MIN
                start_idx += 1
    return PROBLEM_UNBOUND
