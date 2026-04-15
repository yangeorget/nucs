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
    EVENT_MASK_GROUND,
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    MAX,
    MIN,
    NUMBA_DISABLE_JIT,
    PARAM,
    PROBLEM_BOUND,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    RANGE_END,
    RANGE_START,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    TYPE_COMPUTE_DOMAINS,
    VARIABLE,
)
from nucs.heaps import min_heap_pop
from nucs.numba_helper import function_from_address
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, update_propagators_with_previous_prop


@njit(cache=True, fastmath=True)
def bound_consistency_algorithm(
    algorithm_nb: int,
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    bounds: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    decision_variables: NDArray,
) -> int:
    """
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param bounds: the bounds indexed by propagators
    :param propagator_variables: the variables by propagators
    :param propagator_parameters: the parameters by propagators
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :param domains_stk: a stack of domains;
    the first level correspond to the current domains, the rest correspond to the choice points
    :param entailed_propagators_stk: a stack of entailed propagatorspropagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param domain_update_stk: the stack of domain updates, unused here
    :param stks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :param decision_variables: the variables on which decisions will be made
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    top = stks_top[0]
    domains = domains_stk[top]
    entailed_propagators = entailed_propagators_stk[top]
    statistics[STATS_IDX_ALG_BC_NB] += 1
    if NUMBA_DISABLE_JIT:
        compute_domains_fcts = COMPUTE_DOMAINS_FCTS
    else:
        compute_domains_fcts = [function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[0])] * algorithm_nb
        for alg_idx in range(1, algorithm_nb):
            compute_domains_fcts[alg_idx] = function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[alg_idx])
    while True:
        prop_idx = min_heap_pop(triggered_propagators, propagator_nb)
        if prop_idx == -1:
            return PROBLEM_BOUND if unbound_variable_nb_stk[top] == 0 else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = bounds[prop_idx, VARIABLE, RANGE_START]
        prop_var_end = bounds[prop_idx, VARIABLE, RANGE_END]
        prop_domains = domains[propagator_variables[prop_var_start:prop_var_end]]  # this is a copy
        status = compute_domains_fcts[algorithms[prop_idx]](
            prop_domains,
            propagator_parameters[bounds[prop_idx, PARAM, RANGE_START] : bounds[prop_idx, PARAM, RANGE_END]],
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            entailed_propagators[prop_idx] = True
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
        if has_no_changes(
            top,
            propagator_nb,
            prop_idx,
            prop_var_start,
            prop_var_end,
            prop_domains,
            propagator_variables,
            domains,
            triggered_propagators,
            entailed_propagators,
            triggers,
            unbound_variable_nb_stk,
        ):
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1


@njit(cache=True, fastmath=True)
def has_no_changes(
    top: int,
    propagator_nb: int,
    prop_idx: int,
    prop_var_start: int,
    prop_var_end: int,
    prop_domains: NDArray,
    propagator_variables: NDArray,
    domains: NDArray,
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    triggers: NDArray,
    unbound_variable_nb_stk: NDArray,
) -> bool:
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        events = 0
        variable = propagator_variables[prop_var_start + var_idx]
        domain = domains[variable]
        domain_min = prop_domains[var_idx, MIN]
        if domain[MIN] != domain_min:
            domain[MIN] = domain_min
            events |= EVENT_MASK_MIN
        domain_max = prop_domains[var_idx, MAX]
        if domain[MAX] != domain_max:
            domain[MAX] = domain_max
            events |= EVENT_MASK_MAX
        if events:
            if domain_min == domain_max:
                events |= EVENT_MASK_GROUND
                unbound_variable_nb_stk[top] -= 1
            update_propagators_with_previous_prop(
                propagator_nb,
                triggered_propagators,
                entailed_propagators,
                triggers[variable, events],
                prop_idx,
            )
            no_changes = False
    return no_changes
