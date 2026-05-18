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

from nucs.buckets import buckets_add, buckets_pop, STORAGE_OFFSET
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
    EVENT_MASK_NONE,
)


def get_domain_buffer(bounds: NDArray) -> NDArray:
    """
    Allocates a reusable scratch buffer for prop_domains to avoid one allocation per propagator call.

    Sized to the largest propagator arity (which can exceed domain_nb when a propagator
    references the same variable twice, e.g. count_eq).
    Allocated once at solver init and threaded through the consistency algorithms.

    :param bounds: the bounds indexed by propagators
    :type bounds: NDArray

    :return: a scratch buffer sized to the maximal propagator arity
    :rtype: NDArray
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
    propagator_nb: int,
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
    domain_buffer: NDArray,
) -> int:
    """
    This is the default consistency algorithm used by the solver.

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
    :param domains_stk: a stack of domains, the first level correspond to the current domains,
                        the rest correspond to the choice points
    :type domains_stk: NDArray
    :param entailed_propagators_stk: a stack of entailed propagators,
                                     the first level correspond to the propagators currently not entailed,
                                     the rest correspond to the choice points
    :type entailed_propagators_stk: NDArray
    :param domain_update_stk: the stack of domain updates, unused here
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the height of the stacks as a Numpy array
    :type stks_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_fcts: the typed list of compute_domains functions, built once at solver init
    :type compute_domains_fcts: Any
    :param decision_variables: the variables on which decisions will be made
    :type decision_variables: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray

    :return: a status (consistency, inconsistency or entailment) as an integer
    :rtype: int
    """
    top = stks_top[0]
    domains = domains_stk[top]
    entailed_propagators = entailed_propagators_stk[top]
    statistics[STATS_IDX_ALG_BC_NB] += 1
    membership_offset = STORAGE_OFFSET + propagator_nb
    while True:
        prop_idx = buckets_pop(triggered_propagators, membership_offset)
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
        if update_domains(
            top,
            prop_idx,
            prop_var_start,
            prop_var_end,
            membership_offset,
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
def update_domains(
    top: int,
    prop_idx: int,
    prop_var_start: int,
    prop_var_end: int,
    membership_offset: int,
    prop_domains: NDArray,
    propagator_variables: NDArray,
    domains: NDArray,
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    triggers: NDArray,
    unbound_variable_nb_stk: NDArray,
    priorities: NDArray,
) -> bool:
    """
    Updates the domains with the prop_domains computed by a propagator and schedules other propagators triggered by the changes.

    :param top: the index of the top of the stacks
    :type top: int
    :param prop_idx: the index of the propagator that just ran
    :type prop_idx: int
    :param prop_var_start: the start index of the propagator's variables in propagator_variables
    :type prop_var_start: int
    :param prop_var_end: the end index of the propagator's variables in propagator_variables
    :type prop_var_end: int
    :param prop_domains: the domains computed by the propagator
    :type prop_domains: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param domains: the current domains
    :type domains: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param entailed_propagators: the entailed propagators at the current choice point
    :type entailed_propagators: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray

    :return: true iff no domain was changed
    :rtype: bool
    """
    no_changes = True
    # Layout of triggered_propagators (see nucs/buckets.py): the membership flag of propagator p
    # lives at index membership_offset + p. Caching the offset lets us short-circuit buckets_add
    # for propagators already in the queue without paying the function-call overhead.
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
        domain = domains[variable]
        if domain[MIN] != domain[MAX]:
            events = EVENT_MASK_NONE
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
                propagators = triggers[variable, events]
                for other_prop_idx in propagators[1 : propagators[0] + 1]:
                    if (
                        not triggered_propagators[membership_offset + other_prop_idx]
                        and other_prop_idx != prop_idx
                        and not entailed_propagators[other_prop_idx]
                    ):
                        buckets_add(triggered_propagators, priorities, other_prop_idx, membership_offset)
                no_changes = False
    return no_changes
