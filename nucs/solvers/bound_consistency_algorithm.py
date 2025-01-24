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
    PROBLEM_BOUND,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    RG_END,
    RG_START,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    TYPE_COMPUTE_DOMAINS,
)
from nucs.numba_helper import function_from_address
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, pop_propagator, update_propagators
from nucs.solvers.solver import is_solved


@njit(cache=True)
def bound_consistency_algorithm(
    statistics: NDArray,
    no_offsets: bool,
    algorithms: NDArray,
    var_bounds: NDArray,
    param_bounds: NDArray,
    variables_arr: NDArray,
    offsets_arr: NDArray,
    props_variables: NDArray,
    props_offsets: NDArray,
    props_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    decision_domains: NDArray,
) -> int:
    """
    Bound consistency algorithm.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param var_bounds: the variable bounds indexed by propagators
    :param param_bounds: the parameters bounds indexed by propagators
    :param variables_arr: the domain indices indexed by variables, unused here
    :param offsets_arr: the domain offsets indexed by variables, unused here
    :param props_variables: the domain indices indexed by propagator variables
    :param props_offsets: the domain offsets indexed by propagator variables
    :param props_parameters: the parameters indexed by propagator variables
    :param triggers: a Numpy array of event masks indexed by shared domain indices and propagators
    :param domains_stk: a stack of shared domains;
    the first level correspond to the current shared domains, the rest correspond to the choice points
    :param not_entailed_propagators_stk: a stack not entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param dom_update_stk: the stack of domain updates, unused here
    :param stks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :param decision_domains: the indices of the shared domains on which decisions will be made
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    top = stks_top[0]
    statistics[STATS_IDX_ALG_BC_NB] += 1
    prop_idx = -1
    while True:
        prop_idx = pop_propagator(triggered_propagators, prop_idx)
        if prop_idx == -1:
            return PROBLEM_BOUND if is_solved(domains_stk, stks_top) else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = var_bounds[prop_idx, RG_START]
        prop_var_end = var_bounds[prop_idx, RG_END]
        prop_indices = props_variables[prop_var_start:prop_var_end]
        if no_offsets:
            prop_domains = domains_stk[top, prop_indices]
        else:
            prop_offsets = props_offsets[prop_var_start:prop_var_end]
            prop_domains = domains_stk[top, prop_indices] + prop_offsets
        compute_domains_fct = (
            COMPUTE_DOMAINS_FCTS[algorithms[prop_idx]]
            if NUMBA_DISABLE_JIT
            else function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[algorithms[prop_idx]])
        )
        status = compute_domains_fct(
            prop_domains, props_parameters[param_bounds[prop_idx, RG_START] : param_bounds[prop_idx, RG_END]]
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            not_entailed_propagators_stk[top, prop_idx] = False
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
        domains_changes = False
        for var_idx in range(prop_var_end - prop_var_start):
            domain_idx = prop_indices[var_idx]
            events = 0
            if no_offsets:
                domain_min = prop_domains[var_idx, MIN]
                domain_max = prop_domains[var_idx, MAX]
            else:
                offset = prop_offsets[var_idx, 0]  # because of vertical shape
                domain_min = prop_domains[var_idx, MIN] - offset
                domain_max = prop_domains[var_idx, MAX] - offset
            if domains_stk[top, domain_idx, MIN] != domain_min:
                domains_stk[top, domain_idx, MIN] = domain_min
                events |= EVENT_MASK_MIN
            if domains_stk[top, domain_idx, MAX] != domain_max:
                domains_stk[top, domain_idx, MAX] = domain_max
                events |= EVENT_MASK_MAX
            if events != 0 and domain_min == domain_max:
                events |= EVENT_MASK_GROUND
            if events != 0:
                domains_changes = True
                update_propagators(
                    triggered_propagators,
                    not_entailed_propagators_stk[top],
                    triggers,
                    domain_idx,
                    events,
                )
        if not domains_changes:
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
