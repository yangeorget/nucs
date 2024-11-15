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
    MAX,
    MIN,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
    STATS_IDX_ALG_BC_WITH_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_CHANGE_NB,
    STATS_IDX_ALG_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_NO_CHANGE_NB,
)
from nucs.propagators.propagators import add_propagators
from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm
from nucs.solvers.choice_points import backtrack, cp_put
from nucs.solvers.heuristics import (
    first_not_instantiated_var_heuristic_from_index,
    max_value_dom_heuristic,
    min_value_dom_heuristic,
)


@njit(cache=True)
def shave_bound(
    bound: int,
    dom_idx: int,
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
) -> bool:
    if bound == MAX:
        max_value_dom_heuristic(shr_domains_stack, dom_update_stack, stacks_top, dom_idx)
    else:
        min_value_dom_heuristic(shr_domains_stack, dom_update_stack, stacks_top, dom_idx)
    add_propagators(
        triggered_propagators, not_entailed_propagators_stack, stacks_top, shr_domains_propagators, dom_idx, 1 - bound
    )
    if (
        bound_consistency_algorithm(
            statistics,
            algorithms,
            var_bounds,
            param_bounds,
            dom_indices_arr,
            dom_offsets_arr,
            props_dom_indices,
            props_dom_offsets,
            props_parameters,
            shr_domains_propagators,
            shr_domains_stack,
            not_entailed_propagators_stack,
            dom_update_stack,
            stacks_top,
            triggered_propagators,
            compute_domains_addrs,
        )
        == PROBLEM_INCONSISTENT
    ):
        has_shaved = True
    else:
        shr_domains_stack[stacks_top[0] - 1, dom_idx, bound] += 1 if bound == MAX else -1
        has_shaved = False
    backtrack(
        statistics,
        not_entailed_propagators_stack,
        dom_update_stack,
        stacks_top,
        triggered_propagators,
        shr_domains_propagators,
    )
    return has_shaved


@njit(cache=True)
def shaving_consistency_algorithm(
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
    Shaving consistency algorithm.
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
    :param stacks_top: the height of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :return: a status (consistency, inconsistency or entailment) as an integer
    """
    statistics[STATS_IDX_ALG_BC_WITH_SHAVING_NB] += 1
    shr_domains_nb = len(shr_domains_stack[0])
    start_idx = 0
    bound = MIN
    has_shaved = True
    # we could iterate until no domain is shaved, but that would be a lot of extra computation for a small impact
    while start_idx < shr_domains_nb:
        if has_shaved:
            status = bound_consistency_algorithm(
                statistics,
                algorithms,
                var_bounds,
                param_bounds,
                dom_indices_arr,
                dom_offsets_arr,
                props_dom_indices,
                props_dom_offsets,
                props_parameters,
                shr_domains_propagators,
                shr_domains_stack,
                not_entailed_propagators_stack,
                dom_update_stack,
                stacks_top,
                triggered_propagators,
                compute_domains_addrs,
            )
            if status != PROBLEM_UNBOUND:
                return status
        dom_idx = first_not_instantiated_var_heuristic_from_index(shr_domains_stack, stacks_top, start_idx)
        statistics[STATS_IDX_ALG_SHAVING_NB] += 1
        cp_put(shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx)
        has_shaved = shave_bound(
            bound,
            dom_idx,
            statistics,
            algorithms,
            var_bounds,
            param_bounds,
            dom_indices_arr,
            dom_offsets_arr,
            props_dom_indices,
            props_dom_offsets,
            props_parameters,
            shr_domains_propagators,
            shr_domains_stack,
            not_entailed_propagators_stack,
            dom_update_stack,
            stacks_top,
            triggered_propagators,
            compute_domains_addrs,
        )
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
