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

from numba import float64, int32, int64, uint8, uint32

from nucs.solvers.backtrack.bc_algorithm import bc_algorithm

# the built-in consistency algorithm; register_consistency_algorithm adds the others after it
CONSISTENCY_ALG_BC = 0

CONSISTENCY_ALG_FCTS: list[Callable] = [bc_algorithm]  # the functions, by index
CONSISTENCY_ALGS: dict[str, int] = {"BC": CONSISTENCY_ALG_BC}  # name to index, for a selection by name (from the CLI)

SIGN_CONSISTENCY_ALG = int64(
    int64[::1],  # statistics
    float64[::1],  # propagator_weights
    uint8[::1],  # algorithm_flags
    uint8[:, ::1],  # propagator_infos
    uint32[:, ::1],  # offsets
    uint32[::1],  # propagator_variables
    int32[::1],  # propagator_parameters
    int32[::1],  # triggers
    int32[::1],  # triggers_offsets
    int32[::1],  # state
    int32[:, ::1],  # domains, a view of the head of state
    int32[::1],  # entailed, a view of the tail of state
    int32[:, ::1],  # utrail_log
    int32[::1],  # utrail_top
    int32[::1],  # utrail_indices
    int32[:, ::1],  # choicepoints
    uint32[::1],  # choicepoint_top
    int32[::1],  # triggered_propagators
    int64[::1],  # compute_domains_addrs
    int32[:, ::1],  # domain_buffer
)


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


def get_consistency_algorithm_name(consistency_algorithm: int) -> str:
    """
    Returns the name of a consistency algorithm, as the CLI selects it.

    :param consistency_algorithm: the index of the consistency algorithm
    :type consistency_algorithm: int

    :return: the name of the algorithm, or its index when no name refers to it
    :rtype: str
    """
    return next(
        (name for name, idx in CONSISTENCY_ALGS.items() if idx == consistency_algorithm), str(consistency_algorithm)
    )
