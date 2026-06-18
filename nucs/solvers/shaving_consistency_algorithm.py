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
from typing import Any

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


@njit(cache=True, fastmath=True)
def shave_bound(
    algorithm_nb: int,
    propagator_nb: int,
    bound: int,
    variable: int,
    statistics: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
    bounds: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_fcts: Any,
    decision_variables: NDArray,
    domain_buffer: NDArray,
) -> bool:
    """
    Tries to shave one bound of a variable by enforcing bound consistency on the resulting sub-problem.

    :param algorithm_nb: the number of registered propagator algorithms
    :type algorithm_nb: int
    :param bound: the bound to shave (MIN or MAX)
    :type bound: int
    :param variable: the variable whose bound is being shaved
    :type variable: int
    :param statistics: a Numpy array of statistics
    :type statistics: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param bounds: the bounds indexed by propagators
    :type bounds: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the height of the stacks as a Numpy array
    :type stks_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_fcts: the typed list of compute_domains functions
    :type compute_domains_fcts: Any
    :param decision_variables: the variables on which decisions will be made
    :type decision_variables: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray

    :return: true iff the bound was shaved
    :rtype: bool
    """
    events = (
        max_value_dom_heuristic(
            domains_stk,
            entailed_propagator_depths,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            None,  # type: ignore[arg-type]
        )
        if bound == MAX
        else min_value_dom_heuristic(
            domains_stk,
            entailed_propagator_depths,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            None,  # type: ignore[arg-type]
        )
    )
    if domains_stk[stks_top[0], variable, MIN] == domains_stk[stks_top[0], variable, MAX]:
        events |= EVENT_MASK_GROUND
    update_propagators(
        triggered_propagators,
        entailed_propagator_depths,
        triggers[variable, events],
        priorities,
        propagator_nb,
    )
    if (
        bound_consistency_algorithm(
            algorithm_nb,
            propagator_nb,
            statistics,
            algorithms,
            priorities,
            bounds,
            propagator_variables,
            propagator_parameters,
            triggers,
            domains_stk,
            entailed_propagator_depths,
            entailment_trail,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            triggered_propagators,
            compute_domains_fcts,
            decision_variables,
            domain_buffer,
        )
        == PROBLEM_INCONSISTENT
    ):
        has_shaved = True
    else:
        domains_stk[stks_top[0] - 1, variable, bound] += 1 if bound == MAX else -1
        has_shaved = False
    backtrack(
        statistics,
        entailed_propagator_depths,
        entailment_trail,
        domain_update_stk,
        stks_top,
        triggered_propagators,
        triggers,
        priorities,
        propagator_nb,
    )
    return has_shaved


@njit(cache=True, fastmath=True)
def shaving_consistency_algorithm(
    algorithm_nb: int,
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
    bounds: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_fcts: Any,
    decision_variables: NDArray,
    domain_buffer: NDArray,
) -> int:
    """
    This algorithm reduces the need of searching by shaving the domains.

    :param algorithm_nb: the number of registered propagator algorithms
    :type algorithm_nb: int
    :param statistics: a Numpy array of statistics
    :type statistics: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param bounds: the bounds indexed by propagators
    :type bounds: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param domains_stk: a stack of  domains;
                        the first level correspond to the current domains, the rest correspond to the choice points
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the height of the stacks as a Numpy array
    :type stks_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_fcts: the typed list of compute_domains functions
    :type compute_domains_fcts: Any
    :param decision_variables: the variables on which decisions will be made
    :type decision_variables: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray

    :return: a status (consistency, inconsistency or entailment) as an integer
    :rtype: int
    """
    statistics[STATS_IDX_ALG_BC_WITH_SHAVING_NB] += 1
    shr_domains_nb = len(domains_stk[0])
    bound = MIN
    has_shaved = True
    start_idx = 0
    while start_idx < shr_domains_nb:
        if has_shaved:
            status = bound_consistency_algorithm(
                algorithm_nb,
                propagator_nb,
                statistics,
                algorithms,
                priorities,
                bounds,
                propagator_variables,
                propagator_parameters,
                triggers,
                domains_stk,
                entailed_propagator_depths,
                entailment_trail,
                domain_update_stk,
                unbound_variable_nb_stk,
                stks_top,
                triggered_propagators,
                compute_domains_fcts,
                decision_variables,
                domain_buffer,
            )
            if status != PROBLEM_UNBOUND:
                return status
        variable = first_not_instantiated_var_heuristic(
            decision_variables[decision_variables >= start_idx],
            domains_stk,
            stks_top,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )
        if variable == -1:  # all variables after start_idx are instantiated
            break
        statistics[STATS_IDX_ALG_SHAVING_NB] += 1
        has_shaved = shave_bound(
            algorithm_nb,
            propagator_nb,
            bound,
            variable,
            statistics,
            algorithms,
            priorities,
            bounds,
            propagator_variables,
            propagator_parameters,
            triggers,
            domains_stk,
            entailed_propagator_depths,
            entailment_trail,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            triggered_propagators,
            compute_domains_fcts,
            decision_variables,
            domain_buffer,
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
