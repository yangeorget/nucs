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
from nucs.solvers.solver import is_solved


@njit(cache=True)
def bound_consistency_algorithm(
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
    Bound consistency algorithm.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param bounds: the bounds indexed by propagators
    :param props_variables: the variables by propagators
    :param props_parameters: the parameters by propagators
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :param domains_stk: a stack of domains;
    the first level correspond to the current domains, the rest correspond to the choice points
    :param not_entailed_propagators_stk: a stack not entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param dom_update_stk: the stack of domain updates, unused here
    :param stks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :param decision_variables: the variables on which decisions will be made
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    top = stks_top[0]
    statistics[STATS_IDX_ALG_BC_NB] += 1
    if NUMBA_DISABLE_JIT:
        compute_domains_fcts = [COMPUTE_DOMAINS_FCTS[algorithms[prop_idx]] for prop_idx in range(propagator_nb)]
    else:
        previous_algorithm = algorithms[0]
        previous_function = function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[previous_algorithm])
        compute_domains_fcts = [previous_function] * propagator_nb  # necessary for numba to get the type
        for prop_idx in range(1, propagator_nb):
            if algorithms[prop_idx] == previous_algorithm:
                compute_domains_fcts[prop_idx] = previous_function
            else:
                previous_algorithm = algorithms[prop_idx]
                previous_function = compute_domains_fcts[prop_idx] = function_from_address(
                    TYPE_COMPUTE_DOMAINS, compute_domains_addrs[previous_algorithm]
                )
    while True:
        prop_idx = min_heap_pop(triggered_propagators, propagator_nb)
        if prop_idx == -1:
            return PROBLEM_BOUND if is_solved(domains_stk, top) else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = bounds[prop_idx, VARIABLE, RANGE_START]
        prop_var_end = bounds[prop_idx, VARIABLE, RANGE_END]
        prop_domains = domains_stk[top, props_variables[prop_var_start:prop_var_end], :]
        status = compute_domains_fcts[prop_idx](
            prop_domains,
            props_parameters[bounds[prop_idx, PARAM, RANGE_START] : bounds[prop_idx, PARAM, RANGE_END]],
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            not_entailed_propagators_stk[top, prop_idx] = False
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
        no_changes = True
        for var_idx in range(prop_var_end - prop_var_start):
            events = 0
            variable = props_variables[prop_var_start + var_idx]
            domain_min = prop_domains[var_idx, MIN]
            if domains_stk[top, variable, MIN] != domain_min:
                domains_stk[top, variable, MIN] = domain_min
                events |= EVENT_MASK_MIN
            domain_max = prop_domains[var_idx, MAX]
            if domains_stk[top, variable, MAX] != domain_max:
                domains_stk[top, variable, MAX] = domain_max
                events |= EVENT_MASK_MAX
            if events:
                if domain_min == domain_max:
                    events |= EVENT_MASK_GROUND
                update_propagators_with_previous_prop(
                    propagator_nb,
                    triggered_propagators,
                    not_entailed_propagators_stk[top],
                    triggers,
                    events,
                    variable,
                    prop_idx,
                )
                no_changes = False
        if no_changes:
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
