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
# What is left here is what several layers share: the protocols a propagator, a domain heuristic and a
# consistency algorithm are written against, plus the logging and statistics indices. A constant owned by
# one module lives with it instead -- CHOICE_POINT_* in solvers/choice_points.py, OFFSETS_* and PROBLEM_*
# in problems/problem.py, SOLVER_* in solvers/backtrack_solver.py, OPTIM_* in solvers/solver.py, and each
# Numba signature with the registry that compiles against it.
#
# Index constants are prefixed with the array they index -- DOMAIN_, OBJECTIVE_, CHOICE_POINT_, OFFSETS_,
# STATS_IDX_ -- so that a bare index constant never has to be traced back to find out what it indexes.

# Domain bounds
DOMAIN_MIN = 0  # min value of a domain
DOMAIN_MAX = 1  # max value of a domain
DOMAIN_GROUND = 2

# Decision kinds returned by a domain heuristic.
# A domain heuristic chooses where to split a domain; it does not split it. These three kinds cover the
# eight in-tree heuristics exactly, and it is the solver that turns one into an explored branch and one or
# two parked alternatives -- so the min/max/ground bookkeeping lives in one place instead of in every
# heuristic. The parked alternatives are listed deepest first: that order is what an enumeration sees.
DECISION_LE = 0  # explore [min, value], park [value + 1, max]
DECISION_GT = 1  # explore [value + 1, max], park [min, value]
DECISION_EQ = 2  # explore [value, value], park [min, value - 1] then [value + 1, max]

# Statuses returned by a propagator's compute_domains.
PROP_INCONSISTENCY = 0  # returned by a propagator when inconsistent
PROP_CONSISTENCY = 1  # returned by a propagator when consistent
PROP_ENTAILMENT = 2  # returned by a propagator when entailed

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
