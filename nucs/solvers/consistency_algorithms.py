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

from nucs.examples.golomb.golomb_problem import golomb_consistency_algorithm
from nucs.solvers.bc_consistency_algorithm import bound_consistency_algorithm

CONSISTENCY_ALG_BC = 0


@njit(cache=True)  # TODO: fix this
def consistency_algorithm(
    consistency_algorithm_idx: int,
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
    if consistency_algorithm_idx == CONSISTENCY_ALG_BC:
        return bound_consistency_algorithm(
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
            shr_domains_arr,
            not_entailed_propagators,
            triggered_propagators,
            compute_domains_addrs,
        )
    else:
        return golomb_consistency_algorithm(
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
            shr_domains_arr,
            not_entailed_propagators,
            triggered_propagators,
            compute_domains_addrs,
        )
