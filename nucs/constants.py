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
OFFSETS_VARIABLE = 0  # column of offsets holding the propagator variable offsets
OFFSETS_PARAM = 1  # column of offsets holding the propagator parameter offsets

# Domain bounds
DOMAIN_MIN = 0  # min value of a domain
DOMAIN_MAX = 1  # max value of a domain
DOMAIN_GROUND = 2

# Choice point metadata columns.
# One row per choice point, holding what *describes* it rather than what it changed: the trail
# position at its decision point, and the single-bound tightening to apply when the search resumes it.
# None of it is trailed -- trailing the decision would erase the very thing backtrack is about to apply.
CHOICE_POINT_TRAIL_MARK = 0  # the trail size when the choice point branched, the point trail_undo restores to
CHOICE_POINT_VARIABLE = 1  # the variable of the parked alternative
CHOICE_POINT_BOUND = 2  # the side of its domain the alternative tightens
CHOICE_POINT_VALUE = 3  # the value the alternative tightens that side to
CHOICE_POINT_WIDTH = 4  # the number of columns of a choice point

# Capacity outcomes of a search step.
# The trail and the choice point stack are caller-allocated, so they cannot grow inside @njit. Rather than
# sizing them for a worst case that never happens -- depth x (2 x domain_nb + 1) entries, which would
# give back the memory this representation wins -- the search stops and says which one is full, and the
# solver grows it and resumes. Nothing of the search is lost: the state, the trail marks and the
# positions all stay valid across the reallocation.
SOLVER_RUNNING = 0  # nothing filled up: the search returned a solution or exhausted itself
SOLVER_TRAIL_FULL = 1  # the search stopped because the trail needs more room, not because it is over
SOLVER_CHOICE_POINTS_FULL = 2  # likewise for the stack of choice points
SOLVER_STATUS = 0  # index for the status in the status array
SOLVER_STATUS_WIDTH = 1

# Trail entries a step of the search needs beyond one per cell of the backtrackable state.
# The barrier in trail_set trails each cell at most once per choice point, so a fixpoint cannot need more
# than len(state) entries however long it runs. The tightenings the search applies around it are not
# covered by that budget: each writes at a mark the trail holds nothing for yet, so every one of their
# writes is trailed. A tightening writes a domain's two bounds and, when it grounds the variable, the
# unbound count; a step applies at most two of them -- branch's decision is one, while backtracking a
# choice point applies its parked refutation and then the branch-and-bound objective bound.
TIGHTENING_TRAIL_ENTRY_NB = 3  # the two bounds of a domain and the unbound count
STEP_TIGHTENING_NB = 2  # a refutation then the objective bound, the longer of the two ways out of a step

# Decision kinds returned by a domain heuristic.
# A domain heuristic chooses where to split a domain; it does not split it. These three kinds cover the
# eight in-tree heuristics exactly, and it is the solver that turns one into an explored branch and one or
# two parked alternatives -- so the MIN/MAX/GROUND bookkeeping lives in one place instead of in every
# heuristic. The parked alternatives are listed deepest first: that order is what an enumeration sees.
DECISION_LE = 0  # explore [min, value], park [value + 1, max]
DECISION_GT = 1  # explore [value + 1, max], park [min, value]
DECISION_EQ = 2  # explore [value, value], park [min, value - 1] then [value + 1, max]

# Objective indices.
# The branch-and-bound bound is solver state, not choice-point state: it is not backtrackable, so it is
# re-applied to each choice point as the search resumes it rather than written into them all up front.
OBJECTIVE_VARIABLE = 0  # index for the variable being optimized, -1 when not optimizing
OBJECTIVE_BOUND = 1  # index for the side of the optimized domain to tighten
OBJECTIVE_VALUE = 2  # index for the best value found so far
OBJECTIVE_WIDTH = 3  # the number of cells of the objective array

# Events
EVENT_NB = 3
EVENT_MASK_NB = 1 << EVENT_NB
EVENT_MASK_NONE = 0
EVENT_MASK_MIN = 1 << DOMAIN_MIN
EVENT_MASK_MAX = 1 << DOMAIN_MAX
EVENT_MASK_GROUND = 1 << DOMAIN_GROUND
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

# A domain heuristic is a pure decision function: it reads the current domains and returns where to split
# them, as (kind, value). It mutates nothing -- the solver applies the decision.
#
# The pair is int32, and that matters. A heuristic is declared @njit(cache=True) with no signature and
# compiled for this one later, by _get_wrapper_address, so how the pair crosses back is an ABI question.
# Declared as a UniTuple(int64, 2) it is silently wrong: Numba infers `return DECISION_LE,
# domains[variable, MIN]` as the heterogeneous Tuple(int64, int32), and the wrapper *zero-extends* the
# int32, so a split value of -5 arrives as 4294967291. Nothing widens at an int32 pair -- the kind
# narrows from a 0/1/2 constant, losslessly, and the value is already int32 -- so the natural expression
# is correct without the author casting anything. Verified down to INT32_MIN.
SIGN_DOM_HEURISTIC = types.UniTuple(int32, 2)(
    int32[:, ::1],  # domains
    int64,  # variable
    int64[:, :],  # dom_heuristic_params
)
TYPE_DOM_HEURISTIC = types.FunctionType(SIGN_DOM_HEURISTIC)

SIGN_VAR_HEURISTIC = int64(
    uint32[::1],  # decision_variables
    int32[:, ::1],  # domains
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
