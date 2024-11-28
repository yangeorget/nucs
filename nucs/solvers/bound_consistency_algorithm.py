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
# Copyright 2024 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    END,
    MAX,
    MIN,
    NUMBA_DISABLE_JIT,
    PROBLEM_BOUND,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    START,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    TYPE_COMPUTE_DOMAINS,
)
from nucs.numba_helper import function_from_address
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, add_propagators, pop_propagator
from nucs.solvers.solver import is_solved


@njit(cache=True)
def bound_consistency_algorithm(
    statistics: NDArray,
    algorithms: NDArray,
    var_bounds: NDArray,
    param_bounds: NDArray,
    dom_indices_arr: NDArray,
    dom_offsets_arr: NDArray,
    props_dom_indices: NDArray,
    props_dom_offsets: NDArray,
    props_parameters: NDArray,
    shr_domains_propagators: NDArray,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
) -> int:
    """
    Bound consistency algorithm.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param var_bounds: the variable bounds indexed by propagators
    :param param_bounds: the parameters bounds indexed by propagators
    :param dom_indices_arr: the domain indices indexed by variables
    :param dom_offsets_arr: the domain offsets indexed by variables
    :param props_dom_indices: the domain indices indexed by propagator variables
    :param props_dom_offsets: the domain offsets indexed by propagator variables
    :param props_parameters: the parameters indexed by propagator variables
    :param shr_domains_propagators: a Numpy array of booleans indexed
    by shared domain indices, MIN/MAX and propagators; true means that the propagator has to be triggered when the MIN
    or MAX of the shared domain has changed
    :param shr_domains_stack: a stack of shared domains;
    the first level correspond to the current shared domains, the rest correspond to the choice points
    :param not_entailed_propagators_stack: a stack not entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    statistics[STATS_IDX_ALG_BC_NB] += 1
    prop_idx = -1
    while True:
        prop_idx = pop_propagator(triggered_propagators, prop_idx)
        if prop_idx == -1:
            return PROBLEM_BOUND if is_solved(shr_domains_stack, stacks_top) else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = var_bounds[prop_idx, START]
        prop_var_end = var_bounds[prop_idx, END]
        prop_indices = props_dom_indices[prop_var_start:prop_var_end]
        prop_offsets = props_dom_offsets[prop_var_start:prop_var_end]
        prop_domains = shr_domains_stack[stacks_top[0], prop_indices] + prop_offsets
        compute_domains_fct = (
            COMPUTE_DOMAINS_FCTS[algorithms[prop_idx]]
            if NUMBA_DISABLE_JIT
            else function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[algorithms[prop_idx]])
        )
        status = compute_domains_fct(
            prop_domains, props_parameters[param_bounds[prop_idx, START] : param_bounds[prop_idx, END]]
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            not_entailed_propagators_stack[stacks_top[0], prop_idx] = False
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
        shr_domains_changes = False
        top = stacks_top[0]
        for var_idx in range(prop_var_end - prop_var_start):
            shr_domain_idx = prop_indices[var_idx]
            prop_offset = prop_offsets[var_idx, 0]  # because of vertical shape
            for bound in [MIN, MAX]:
                shr_domain_bound = prop_domains[var_idx, bound] - prop_offset
                if shr_domains_stack[top, shr_domain_idx, bound] != shr_domain_bound:
                    shr_domains_stack[top, shr_domain_idx, bound] = shr_domain_bound
                    shr_domains_changes = True
                    add_propagators(
                        triggered_propagators,
                        not_entailed_propagators_stack,
                        stacks_top,
                        shr_domains_propagators,
                        shr_domain_idx,
                        bound,
                    )
        if not shr_domains_changes:
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
