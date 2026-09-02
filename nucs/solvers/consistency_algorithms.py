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
from collections.abc import Callable

from numba import boolean, int32, int64, types, uint8, uint32

from nucs.propagators.propagators import TYPE_COMPUTE_DOMAINS_LIST
from nucs.solvers.bc_algorithm import bc_algorithm

CONSISTENCY_ALG_FCTS: list[Callable] = []
CONSISTENCY_ALGS: dict[str, int] = {}  # algorithm name to index, for name-based selection (eg from the CLI)

SIGN_CONSISTENCY_ALG = int64(
    int64[::1],  # statistics
    boolean[::1],  # idempotencies
    uint8[::1],  # algorithms
    uint32[::1],  # priorities
    uint32[:, ::1],  # offsets
    uint32[::1],  # propagator_variables
    int32[::1],  # propagator_parameters
    int32[::1],  # triggers
    int32[::1],  # triggers_offsets
    int32[::1],  # state
    int32[:, ::1],  # domains, a view of the head of state
    int32[::1],  # entailed, a view of the tail of state
    int32[:, ::1],  # trail
    int32[::1],  # trail_top
    int32[::1],  # pos
    int32[:, ::1],  # choice_point_stk
    uint32[::1],  # choice_point_top
    int32[::1],  # triggered_propagators
    TYPE_COMPUTE_DOMAINS_LIST,  # compute_domains_fcts
    int32[:, ::1],  # domain_buffer
)
TYPE_CONSISTENCY_ALG = types.FunctionType(SIGN_CONSISTENCY_ALG)


def register_consistency_algorithm(consistency_algorithm_fct: Callable, name: str | None = None) -> int:
    """
    Register a consistency algorithm by adding its function to the corresponding list of functions.

    :param consistency_algorithm_fct: a function that enforces consistency
    :type consistency_algorithm_fct: Callable
    :param name: the name of the algorithm,
                 defaults to the function name without its _consistency_algorithm suffix, uppercased
    :type name: Optional[str]

    :return: the index of the consistency algorithm
    :rtype: int
    """
    CONSISTENCY_ALG_FCTS.append(consistency_algorithm_fct)
    if name is None:
        name = consistency_algorithm_fct.__name__.removesuffix("_consistency_algorithm").upper()
    CONSISTENCY_ALGS[name] = len(CONSISTENCY_ALG_FCTS) - 1
    return len(CONSISTENCY_ALG_FCTS) - 1


CONSISTENCY_ALG_BC = register_consistency_algorithm(bc_algorithm, "BC")
