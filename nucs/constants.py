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
import os

from numba import bool, int32, int64, types, uint8, uint16  # type: ignore

START = 0  # index corresponding the start of a values range
END = 1  # index corresponding the end of a values range

MIN = 0  # min value of a domain
MAX = 1  # max value of a domain

PROP_INCONSISTENCY = 0  # returned by a propagator when inconsistent
PROP_CONSISTENCY = 1  # returned by a propagator when consistent
PROP_ENTAILMENT = 2  # returned by a propagator when entailed

PROBLEM_INCONSISTENT = 0  # returned when the filtering of a problem detects an inconsistency
PROBLEM_UNBOUND = 1  # returned when the filtering of a problem has been completed but the problem is not solved
PROBLEM_BOUND = 2  # returned when a problem is solved

SIGNATURE_COMPUTE_DOMAINS = int64(int32[:, :], int32[:])  # domains, parameters
SIGNATURE_DOM_HEURISTIC = int64(
    int32[:, :, :], uint16[:, :], uint8[:], int64  # shr_domains_stack  # dom_update_stack  # stacks_top  # dom_idx
)
SIGNATURE_VAR_HEURISTIC = int64(int32[:, :, :], uint8[:])  # shr_domains_stack, stacks_top
SIGNATURE_CONSISTENCY_ALG = int64(
    int64[:],  # statistics
    uint8[:],  # algorithms
    uint16[:, :],  # var_bounds
    uint16[:, :],  # param_bounds
    uint16[:],  # dom_indices_arr
    int32[:],  # dom_offsets_arr
    uint16[:],  # props_dom_indices
    int32[:, :],  # props_dom_offsets
    int32[:],  # props_parameters
    bool[:, :, :],  # shr_domains_propagators
    int32[:, :, :],  # shr_domains_stack
    bool[:, :],  # not_entailed_propagators_stack
    uint16[:, :],  # dom_update_stack
    uint8[:],  # stacks_height
    bool[:],  # triggered_propagators
    int64[:],  # compute_domains_addrs
)

TYPE_COMPUTE_DOMAINS = types.FunctionType(SIGNATURE_COMPUTE_DOMAINS)
TYPE_DOM_HEURISTIC = types.FunctionType(SIGNATURE_DOM_HEURISTIC)
TYPE_VAR_HEURISTIC = types.FunctionType(SIGNATURE_VAR_HEURISTIC)
TYPE_CONSISTENCY_ALG = types.FunctionType(SIGNATURE_CONSISTENCY_ALG)

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT")

LOG_FORMAT = "%(asctime)s - %(processName)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"
LOG_LEVELS = [LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, LOG_LEVEL_ERROR, LOG_LEVEL_CRITICAL]

STATS_MAX = 13
(
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_ALG_BC_WITH_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_CHANGE_NB,
    STATS_IDX_ALG_SHAVING_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_SOLUTION_NB,
) = tuple(range(STATS_MAX))

STATS_LBL_ALG_BC_NB = "ALG_BC_NB"
STATS_LBL_ALG_BC_WITH_SHAVING_NB = "ALG_BC_WITH_SHAVING_NB"
STATS_LBL_ALG_SHAVING_NB = "ALG_SHAVING_NB"
STATS_LBL_ALG_SHAVING_CHANGE_NB = "ALG_SHAVING_CHANGE_NB"
STATS_LBL_ALG_SHAVING_NO_CHANGE_NB = "ALG_SHAVING_NO_CHANGE_NB"
STATS_LBL_PROPAGATOR_ENTAILMENT_NB = "PROPAGATOR_ENTAILMENT_NB"
STATS_LBL_PROPAGATOR_FILTER_NB = "PROPAGATOR_FILTER_NB"
STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB = "PROPAGATOR_FILTER_NO_CHANGE_NB"
STATS_LBL_PROPAGATOR_INCONSISTENCY_NB = "PROPAGATOR_INCONSISTENCY_NB"
STATS_LBL_SOLVER_BACKTRACK_NB = "SOLVER_BACKTRACK_NB"
STATS_LBL_SOLVER_CHOICE_NB = "SOLVER_CHOICE_NB"
STATS_LBL_SOLVER_CHOICE_DEPTH = "SOLVER_CHOICE_DEPTH"
STATS_LBL_SOLVER_SOLUTION_NB = "SOLVER_SOLUTION_NB"
