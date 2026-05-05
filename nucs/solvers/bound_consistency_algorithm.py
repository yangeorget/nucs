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

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_pop
from nucs.constants import (
    EVENT_MASK_GROUND,
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    MAX,
    MIN,
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
    VARIABLE,
)
from nucs.propagators.propagators import update_propagators_with_previous_prop


@njit(cache=True, fastmath=True)
def get_domain_buffer(bounds: NDArray) -> NDArray:
    """
    Reusable scratch buffer for prop_domains: avoids one allocation per propagator call.
    Sized to the largest propagator arity (which can exceed domain_nb when a propagator
    references the same variable twice, e.g. count_eq).
    """
    max_arity = np.int64(0)
    for propagator_idx in range(len(bounds)):
        arity = np.int64(bounds[propagator_idx, VARIABLE, RANGE_END] - bounds[propagator_idx, VARIABLE, RANGE_START])
        if arity > max_arity:
            max_arity = arity
    return np.empty((max_arity, 2), dtype=np.int32)


@njit(cache=True, fastmath=True)
def bound_consistency_algorithm(
    algorithm_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
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
    compute_domains_fcts: Any,
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
    :param entailed_propagators_stk: a stack of entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param domain_update_stk: the stack of domain updates, unused here
    :param stks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_fcts: the typed list of compute_domains functions, built once at solver init
    :param decision_variables: the variables on which decisions will be made
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    top = stks_top[0]
    domains = domains_stk[top]
    entailed_propagators = entailed_propagators_stk[top]
    statistics[STATS_IDX_ALG_BC_NB] += 1
    domain_buffer = get_domain_buffer(bounds)
    while True:
        prop_idx = buckets_pop(triggered_propagators)
        if prop_idx == -1:
            return PROBLEM_BOUND if unbound_variable_nb_stk[top] == 0 else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = bounds[prop_idx, VARIABLE, RANGE_START]
        prop_var_end = bounds[prop_idx, VARIABLE, RANGE_END]
        prop_arity = prop_var_end - prop_var_start
        prop_domains = domain_buffer[:prop_arity]
        for var_idx in range(prop_arity):
            prop_domains[var_idx] = domains[propagator_variables[prop_var_start + var_idx]]
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
            priorities,
        ):
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1


@njit(cache=True, fastmath=True)
def has_no_changes(
    top: int,
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
    priorities: NDArray,
) -> bool:
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
        domain = domains[variable]
        if domain[MIN] != domain[MAX]:
            events = 0
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
                    triggered_propagators, entailed_propagators, triggers[variable, events], prop_idx, priorities
                )
                no_changes = False
    return no_changes
