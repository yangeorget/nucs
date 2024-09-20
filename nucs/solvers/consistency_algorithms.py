import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    END,
    MAX,
    MIN,
    PROBLEM_FILTERED,
    PROBLEM_INCONSISTENT,
    PROBLEM_SOLVED,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    START,
)
from nucs.numba import NUMBA_DISABLE_JIT, function_from_address
from nucs.problems.problem import Problem, is_solved
from nucs.propagators.propagators import (
    COMPUTE_DOMAIN_TYPE,
    COMPUTE_DOMAINS_ADDRS,
    COMPUTE_DOMAINS_FCTS,
    pop_propagator,
)
from nucs.statistics import (
    STATS_PROBLEM_FILTER_NB,
    STATS_PROPAGATOR_ENTAILMENT_NB,
    STATS_PROPAGATOR_FILTER_NB,
    STATS_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_PROPAGATOR_INCONSISTENCY_NB,
)


def bound_consistency_algorithm(statistics: NDArray, problem: Problem) -> int:
    return _bound_consistency_algorithm(
        statistics,
        problem.algorithms,
        problem.var_bounds,
        problem.data_bounds,
        problem.props_dom_indices,
        problem.props_dom_offsets,
        problem.props_data,
        problem.shr_domains_arr,
        problem.shr_domains_propagators,
        problem.triggered_propagators,
        problem.not_entailed_propagators,
        COMPUTE_DOMAINS_ADDRS,
    )


@njit(cache=True)
def _bound_consistency_algorithm(
    statistics: NDArray,
    algorithms: NDArray,
    var_bounds: NDArray,
    data_bounds: NDArray,
    props_indices: NDArray,
    props_offsets: NDArray,
    props_data: NDArray,
    shr_domains: NDArray,
    shr_domains_propagators: NDArray,
    triggered_propagators: NDArray,
    not_entailed_propagators: NDArray,
    compute_domains_addrs: NDArray,
) -> int:
    """
    Filters the problem's domains by applying the propagators until a fix point is reached.
    """
    statistics[STATS_PROBLEM_FILTER_NB] += 1
    prop_idx = -1
    while True:
        prop_idx = pop_propagator(triggered_propagators, not_entailed_propagators, prop_idx)
        if prop_idx == -1:
            return PROBLEM_SOLVED if is_solved(shr_domains) else PROBLEM_FILTERED
        statistics[STATS_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = var_bounds[prop_idx, START]
        prop_var_end = var_bounds[prop_idx, END]
        prop_indices = props_indices[prop_var_start:prop_var_end]
        prop_offsets = props_offsets[prop_var_start:prop_var_end]
        prop_var_nb = prop_var_end - prop_var_start
        prop_domains = np.empty((2, prop_var_nb), dtype=np.int32).T  # trick for order=F
        np.add(shr_domains[prop_indices], prop_offsets, prop_domains)
        algorithm = algorithms[prop_idx]
        compute_domains_function = (
            COMPUTE_DOMAINS_FCTS[algorithm]
            if NUMBA_DISABLE_JIT
            else function_from_address(COMPUTE_DOMAIN_TYPE, compute_domains_addrs[algorithm])
        )
        prop_data = props_data[data_bounds[prop_idx, START] : data_bounds[prop_idx, END]]
        status = compute_domains_function(prop_domains, prop_data)
        if status == PROP_INCONSISTENCY:
            statistics[STATS_PROPAGATOR_INCONSISTENCY_NB] += 1
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            not_entailed_propagators[prop_idx] = False
            statistics[STATS_PROPAGATOR_ENTAILMENT_NB] += 1
        shr_domains_changes = False
        for var_idx in range(prop_var_nb):
            shr_domain_idx = prop_indices[var_idx]
            prop_offset = prop_offsets[var_idx][0]
            shr_domain_min = prop_domains[var_idx, MIN] - prop_offset
            shr_domain_max = prop_domains[var_idx, MAX] - prop_offset
            if shr_domains[shr_domain_idx, MIN] != shr_domain_min:
                shr_domains[shr_domain_idx, MIN] = shr_domain_min
                shr_domains_changes = True
                np.logical_or(
                    triggered_propagators, shr_domains_propagators[shr_domain_idx, MIN], triggered_propagators
                )
            if shr_domains[shr_domain_idx, MAX] != shr_domain_max:
                shr_domains[shr_domain_idx, MAX] = shr_domain_max
                shr_domains_changes = True
                np.logical_or(
                    triggered_propagators, shr_domains_propagators[shr_domain_idx, MAX], triggered_propagators
                )
        if not shr_domains_changes:  # type: ignore
            statistics[STATS_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
