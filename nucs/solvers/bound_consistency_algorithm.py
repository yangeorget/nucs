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
import numpy as np
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
    TYPE_COMPUTE_DOMAINS,
)
from nucs.numba_helper import function_from_address
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, pop_propagator
from nucs.solvers.solver import is_solved
from nucs.statistics import (
    STATS_IDX_PROBLEM_FILTER_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
)


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
    shr_domains_arr: NDArray,
    not_entailed_propagators: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
) -> int:
    """
    Internal method for applying the bound consistency algorithm.
    This method only uses Numpy arrays as parameters, this permits JIT compilation.
    """
    statistics[STATS_IDX_PROBLEM_FILTER_NB] += 1
    prop_idx = -1
    while True:
        prop_idx = pop_propagator(triggered_propagators, not_entailed_propagators, prop_idx)
        if prop_idx == -1:
            return PROBLEM_BOUND if is_solved(shr_domains_arr) else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = var_bounds[prop_idx, START]
        prop_var_end = var_bounds[prop_idx, END]
        prop_indices = props_dom_indices[prop_var_start:prop_var_end]
        prop_offsets = props_dom_offsets[prop_var_start:prop_var_end]
        prop_var_nb = prop_var_end - prop_var_start
        prop_domains = np.empty((prop_var_nb, 2), dtype=np.int32)
        np.add(shr_domains_arr[prop_indices], prop_offsets, prop_domains)
        compute_domains_function = (
            COMPUTE_DOMAINS_FCTS[algorithms[prop_idx]]
            if NUMBA_DISABLE_JIT
            else function_from_address(TYPE_COMPUTE_DOMAINS, compute_domains_addrs[algorithms[prop_idx]])
        )
        status = compute_domains_function(
            prop_domains, props_parameters[param_bounds[prop_idx, START] : param_bounds[prop_idx, END]]
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            not_entailed_propagators[prop_idx] = False
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
        shr_domains_changes = False
        for var_idx in range(prop_var_nb):
            shr_domain_idx = prop_indices[var_idx]
            prop_offset = prop_offsets[var_idx, 0]
            for bound in [MIN, MAX]:
                shr_domain_bound = prop_domains[var_idx, bound] - prop_offset
                if shr_domains_arr[shr_domain_idx, bound] != shr_domain_bound:
                    shr_domains_arr[shr_domain_idx, bound] = shr_domain_bound
                    shr_domains_changes = True
                    np.logical_or(
                        triggered_propagators, shr_domains_propagators[shr_domain_idx, bound], triggered_propagators
                    )
        if not shr_domains_changes:
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
