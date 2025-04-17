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
import os

from numba import bool, int32, int64, types, uint8, uint16, uint32, uint64  # type: ignore

# Progress bar modes
PB_NONE = 0
PB_SLAVE = 1
PB_MASTER = 2
PB_BLOCK_NB = 1 << 16


# Optimizer modes
OPTIM_RESET = "RESET"
OPTIM_PRUNE = "PRUNE"
OPTIM_MODES = [OPTIM_RESET, OPTIM_PRUNE]


# Bounds
VARIABLE = 0  # index for a variable
PARAM = 1  # index for a parameter
RANGE_START = 0  # index corresponding to the start of a value range
RANGE_END = 1  # index corresponding to the end of a value range


# Domain bounds
MIN = 0  # min value of a domain
MAX = 1  # max value of a domain


# Domain update stack indices
DOM_UPDATE_IDX = 0  # index for the domain idx
DOM_UPDATE_EVENTS = 1  # index for the event


# Events
EVENT_NB = 3
EVENT_MASK_MIN = 1 << 0
EVENT_MASK_MAX = 1 << 1
EVENT_MASK_GROUND = 1 << 2
EVENT_MASK_MIN_MAX = EVENT_MASK_MIN | EVENT_MASK_MAX
EVENT_MASK_MIN_GROUND = EVENT_MASK_MIN | EVENT_MASK_GROUND
EVENT_MASK_MAX_GROUND = EVENT_MASK_MAX | EVENT_MASK_GROUND
EVENT_MASK_MIN_MAX_GROUND = EVENT_MASK_MIN | EVENT_MASK_MAX | EVENT_MASK_GROUND


PROP_INCONSISTENCY = 0  # returned by a propagator when inconsistent
PROP_CONSISTENCY = 1  # returned by a propagator when consistent
PROP_ENTAILMENT = 2  # returned by a propagator when entailed

PROBLEM_INCONSISTENT = 0  # returned when the filtering of a problem detects an inconsistency
PROBLEM_UNBOUND = 1  # returned when the filtering of a problem has been completed, but the problem is not solved
PROBLEM_BOUND = 2  # returned when a problem is solved

SIGNATURE_COMPUTE_DOMAINS = int64(int32[:, :], int32[:])  # domains, parameters
SIGNATURE_GET_TRIGGERS = int64(uint64, uint64, int32[:])
SIGNATURE_CONSISTENCY_ALG = int64(
    int64[:],  # statistics
    uint8[:],  # algorithms
    uint32[:, :, :],  # bounds
    uint32[:],  # variables_arr
    int32[:],  # offsets_arr
    uint32[:],  # props_variables
    int32[:],  # props_offsets
    int32[:],  # props_parameters
    int32[:, :, :],  # triggers
    int32[:, :, :],  # domains_stk
    bool[:, :],  # not_entailed_propagators_stk
    uint32[:, :],  # dom_update_stk
    uint16[:],  # stks_top
    uint32[:],  # triggered_propagators
    int64[:],  # compute_domains_addrs
    uint32[:],  # decision_domains
)
SIGNATURE_DOM_HEURISTIC = int64(
    int32[:, :, :],  # domains_stk
    bool[:, :],  # not_entailed_propagators_stk
    uint32[:, :],  # dom_update_stk
    uint16[:],  # stks_top
    int64,  # dom_idx
    int64[:, :],  # dom_heuristic_params
)
SIGNATURE_VAR_HEURISTIC = int64(
    uint32[:],  # decision_variables
    int32[:, :, :],  # domains_stk
    uint16[:],  # stks_top
    int64[:, :],  # var_heuristic_params
)


TYPE_COMPUTE_DOMAINS = types.FunctionType(SIGNATURE_COMPUTE_DOMAINS)
TYPE_GET_TRIGGERS = types.FunctionType(SIGNATURE_GET_TRIGGERS)
TYPE_DOM_HEURISTIC = types.FunctionType(SIGNATURE_DOM_HEURISTIC)
TYPE_VAR_HEURISTIC = types.FunctionType(SIGNATURE_VAR_HEURISTIC)
TYPE_CONSISTENCY_ALG = types.FunctionType(SIGNATURE_CONSISTENCY_ALG)

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT")

LOG_FORMAT = "[ %(asctime)s | %(processName)s | %(levelname)s ] %(name)s.%(funcName)s - %(message)s"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"
LOG_LEVELS = [LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, LOG_LEVEL_ERROR, LOG_LEVEL_CRITICAL]

STATS_MAX = 16
(
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_ALG_BC_WITH_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_CHANGE_NB,
    STATS_IDX_ALG_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SEARCH_SPACE_INITIAL_SZ,
    STATS_IDX_SEARCH_SPACE_REMAINING_SZ,
    STATS_IDX_SEARCH_SPACE_LOG2_SCALE,
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
) = tuple(range(STATS_MAX))

STATS_LBL_ALG_BC_NB = "ALG_BC_NB"
STATS_LBL_ALG_BC_WITH_SHAVING_NB = "ALG_BC_WITH_SHAVING_NB"
STATS_LBL_ALG_SHAVING_CHANGE_NB = "ALG_SHAVING_CHANGE_NB"
STATS_LBL_ALG_SHAVING_NB = "ALG_SHAVING_NB"
STATS_LBL_ALG_SHAVING_NO_CHANGE_NB = "ALG_SHAVING_NO_CHANGE_NB"
STATS_LBL_PROPAGATOR_ENTAILMENT_NB = "PROPAGATOR_ENTAILMENT_NB"
STATS_LBL_PROPAGATOR_FILTER_NB = "PROPAGATOR_FILTER_NB"
STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB = "PROPAGATOR_FILTER_NO_CHANGE_NB"
STATS_LBL_PROPAGATOR_INCONSISTENCY_NB = "PROPAGATOR_INCONSISTENCY_NB"
STATS_LBL_SEARCH_SPACE_INITIAL_SZ = "SEARCH_SPACE_INITIAL_SZ"
STATS_LBL_SEARCH_SPACE_REMAINING_SZ = "SEARCH_SPACE_REMAINING_SZ"
STATS_LBL_SEARCH_SPACE_LOG2_SCALE = "SEARCH_SPACE_LOG2_SCALE"
STATS_LBL_SOLUTION_NB = "SOLUTION_NB"
STATS_LBL_SOLVER_BACKTRACK_NB = "SOLVER_BACKTRACK_NB"
STATS_LBL_SOLVER_CHOICE_DEPTH = "SOLVER_CHOICE_DEPTH"
STATS_LBL_SOLVER_CHOICE_NB = "SOLVER_CHOICE_NB"
