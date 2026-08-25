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
import os

from numba import boolean, int32, int64, types, uint8, uint32, uint64  # type: ignore

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
GROUND = 2

# Domain update stack indices
DOM_UPDATE_VARIABLE = 0  # index for the variable
DOM_UPDATE_EVENTS = 1  # index for the events

# Events
EVENT_NB = 3
EVENT_MASK_NB = 1 << EVENT_NB
EVENT_MASK_NONE = 0
EVENT_MASK_MIN = 1 << MIN
EVENT_MASK_MAX = 1 << MAX
EVENT_MASK_GROUND = 1 << GROUND
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

# The array arguments are typed C-contiguous (::1) rather than any-layout (:) so the hot loops in every
# propagator and in the consistency algorithm index with a plain offset instead of a stride multiply.
# All these arrays are contiguous np.empty/np.zeros/np.ones allocations threaded through unchanged.
SIGN_COMPUTE_DOMAINS = int64(int32[:, ::1], int32[::1])  # domains, parameters
TYPE_COMPUTE_DOMAINS = types.FunctionType(SIGN_COMPUTE_DOMAINS)
TYPE_COMPUTE_DOMAINS_LIST = types.ListType(TYPE_COMPUTE_DOMAINS)

SIGN_GET_TRIGGERS = int64(uint64, uint64, int32[::1])
TYPE_GET_TRIGGERS = types.FunctionType(SIGN_GET_TRIGGERS)

SIGN_CONSISTENCY_ALG = int64(
    int64,  # propagator_nb
    int64[::1],  # statistics
    uint8[::1],  # algorithms
    uint32[::1],  # complexities
    uint32[:, :, ::1],  # bounds
    uint32[::1],  # propagator_variables
    int32[::1],  # propagator_parameters
    int32[::1],  # triggers
    int32[::1],  # triggers_offsets
    int32[:, :, ::1],  # domains_stk
    int32[::1],  # entailed_propagator_depths
    int32[::1],  # entailment_trail
    uint32[::1],  # unbound_variable_nb_stk
    uint32[::1],  # stks_top
    int32[::1],  # triggered_propagators
    TYPE_COMPUTE_DOMAINS_LIST,  # compute_domains_fcts
    int32[:, ::1],  # domain_buffer
    boolean[::1],  # idempotent
)
TYPE_CONSISTENCY_ALG = types.FunctionType(SIGN_CONSISTENCY_ALG)

SIGN_DOM_HEURISTIC = int64(
    int32[:, :, :],  # domains_stk
    uint32[:, :],  # domain_update_stk
    uint32[:],  # unbound_variable_nb_stk
    uint32[:],  # stks_top
    int64,  # variable
    int64[:, :],  # dom_heuristic_params
)
TYPE_DOM_HEURISTIC = types.FunctionType(SIGN_DOM_HEURISTIC)

SIGN_VAR_HEURISTIC = int64(
    uint32[:],  # decision_variables
    int32[:, :, :],  # domains_stk
    int64,  # top
    int64[:, :],  # var_heuristic_params
)
TYPE_VAR_HEURISTIC = types.FunctionType(SIGN_VAR_HEURISTIC)

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT")

LOG_FORMAT = "[ %(asctime)s | %(processName)s | %(levelname)s ] %(name)s.%(funcName)s - %(message)s"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"
LOG_LEVELS = [LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, LOG_LEVEL_ERROR, LOG_LEVEL_CRITICAL]

STATS_MAX = 10
(
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_ELAPSED_TIME,
) = tuple(range(STATS_MAX))

# The statistics array carries a per-algorithm tail after the STATS_MAX global counters: two counters per
# registered algorithm, at STATS_MAX + STATS_ALG_WIDTH * algorithm. It rides along in the same array so the
# jitted consistency-algorithm signature does not have to grow a parameter for it.
STATS_ALG_WIDTH = 2
STATS_ALG_IDX_FILTER_NB = 0  # calls made
STATS_ALG_IDX_FILTER_NO_CHANGE_NB = 1  # calls that pruned nothing

STATS_LBL_ALG_BC_NB = "ALG_BC_NB"
STATS_LBL_PROPAGATOR_ENTAILMENT_NB = "PROPAGATOR_ENTAILMENT_NB"
STATS_LBL_PROPAGATOR_FILTER_NB = "PROPAGATOR_FILTER_NB"
STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB = "PROPAGATOR_FILTER_NO_CHANGE_NB"
STATS_LBL_PROPAGATOR_INCONSISTENCY_NB = "PROPAGATOR_INCONSISTENCY_NB"
STATS_LBL_SOLUTION_NB = "SOLUTION_NB"
STATS_LBL_SOLVER_BACKTRACK_NB = "SOLVER_BACKTRACK_NB"
STATS_LBL_SOLVER_CHOICE_DEPTH = "SOLVER_CHOICE_DEPTH"
STATS_LBL_SOLVER_CHOICE_NB = "SOLVER_CHOICE_NB"
STATS_LBL_SOLVER_ELAPSED_TIME = "SOLVER_ELAPSED_TIME_MS"
