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
"""
The statistics array: its layout, the labels it is reported under, and how it is read back.

One int64 array carries everything the solver counts, so that a jitted function can bump a counter with a
single store and the whole of it crosses into nopython mode as one argument.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

STATS_MAX = 11
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
    STATS_IDX_SOLVER_RESTART_NB,
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
STATS_LBL_SOLVER_RESTART_NB = "SOLVER_RESTART_NB"

# The statistics of the learning of LcgSolver, in an array of their own, so that the array of BacktrackSolver and its
# propagation loop do not change. The global counters come first, then one cell for each registered algorithm, at
# STATS_LCG_MAX + algorithm: the explanations that the conflict analysis asked of the propagators of the algorithm.
# The explanation counters measure the work of the analysis: they include the explanations of an analysis that gives
# no asserting clause and starts again on the decisions of the path.
# The means are not counters: the mean length of the learned clauses is STATS_LCG_IDX_LEARNED_LIT_NB divided by
# STATS_LCG_IDX_CONFLICT_NB, and the mean LBD is STATS_LCG_IDX_LEARNED_LBD_SUM divided by STATS_LCG_IDX_CONFLICT_NB.
# The empty clause, at most one at the end of a search, counts as a conflict with no literal and an LBD of 0.
STATS_LCG_MAX = 5
(
    # the levels that the backjumps removed, from the level of the search, not of the conflict: chronological
    # backtracking would remove 1 for each conflict
    STATS_LCG_IDX_BACKJUMP_LEVEL_NB,
    STATS_LCG_IDX_CLAUSE_EXPLANATION_NB,  # the explanations that the conflict analysis asked of the learned clauses
    STATS_LCG_IDX_CONFLICT_NB,  # the conflicts that the analysis turned into a clause, the nogoods of solutions included
    STATS_LCG_IDX_LEARNED_LBD_SUM,  # the sum of the LBDs of the learned clauses
    STATS_LCG_IDX_LEARNED_LIT_NB,  # the literals of the learned clauses
) = tuple(range(STATS_LCG_MAX))

STATS_LCG_LBL_BACKJUMP_LEVEL_NB = "LCG_BACKJUMP_LEVEL_NB"
STATS_LCG_LBL_CLAUSE_EXPLANATION_NB = "LCG_CLAUSE_EXPLANATION_NB"
STATS_LCG_LBL_CONFLICT_NB = "LCG_CONFLICT_NB"
STATS_LCG_LBL_LEARNED_LBD_SUM = "LCG_LEARNED_LBD_SUM"
STATS_LCG_LBL_LEARNED_LIT_NB = "LCG_LEARNED_LIT_NB"
STATS_LCG_LBL_PROPAGATOR_EXPLANATION_NB = "LCG_PROPAGATOR_EXPLANATION_NB"


def statistics_init(algorithm_nb: int) -> NDArray:
    """
    Allocates the statistics array: the global counters, then two per registered algorithm.

    :param algorithm_nb: the number of registered propagator algorithms
    :type algorithm_nb: int

    :return: the statistics array, zeroed
    :rtype: NDArray
    """
    return np.zeros(STATS_MAX + STATS_ALG_WIDTH * algorithm_nb, dtype=np.int64)


def statistics_as_dictionary(statistics: NDArray, algorithm_names: Sequence[str]) -> dict[str, int]:
    """
    Returns the statistics as a dictionary, labelled and with the per-algorithm tail broken out.

    Beyond the global counters, the dictionary breaks the two filtering counters down per propagator
    algorithm, restricted to the algorithms that ran at least once so that the breakdown stays readable.
    Each entry is suffixed with the algorithm name, so a breakdown sorts next to the total it partitions.

    A call that prunes nothing still costs a bucket pop, a gather of its variables' domains into the
    scratch buffer, an indirect call and a write-back, so a high no-change share on a given algorithm is
    where wasted propagation is concentrated.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param algorithm_names: the display name of each registered algorithm, indexed by algorithm
    :type algorithm_names: Sequence[str]

    :return: a dictionary mapping statistic labels to values
    :rtype: Dict[str, int]
    """
    dictionary = {
        STATS_LBL_ALG_BC_NB: int(statistics[STATS_IDX_ALG_BC_NB]),
        STATS_LBL_PROPAGATOR_ENTAILMENT_NB: int(statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB]),
        STATS_LBL_PROPAGATOR_FILTER_NB: int(statistics[STATS_IDX_PROPAGATOR_FILTER_NB]),
        STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB: int(statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB]),
        STATS_LBL_PROPAGATOR_INCONSISTENCY_NB: int(statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB]),
        STATS_LBL_SOLVER_BACKTRACK_NB: int(statistics[STATS_IDX_SOLVER_BACKTRACK_NB]),
        STATS_LBL_SOLVER_CHOICE_NB: int(statistics[STATS_IDX_SOLVER_CHOICE_NB]),
        STATS_LBL_SOLVER_CHOICE_DEPTH: int(statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]),
        STATS_LBL_SOLVER_RESTART_NB: int(statistics[STATS_IDX_SOLVER_RESTART_NB]),
        STATS_LBL_SOLUTION_NB: int(statistics[STATS_IDX_SOLUTION_NB]),
        # the statistics array accumulates nanoseconds, the reported statistic is in milliseconds
        STATS_LBL_SOLVER_ELAPSED_TIME: int(statistics[STATS_IDX_SOLVER_ELAPSED_TIME]) // 1_000_000,
    }
    for algorithm, name in enumerate(algorithm_names):
        base = STATS_MAX + STATS_ALG_WIDTH * algorithm
        calls = int(statistics[base + STATS_ALG_IDX_FILTER_NB])
        if calls:
            dictionary[f"{STATS_LBL_PROPAGATOR_FILTER_NB}_{name}"] = calls
            dictionary[f"{STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB}_{name}"] = int(
                statistics[base + STATS_ALG_IDX_FILTER_NO_CHANGE_NB]
            )
    return dictionary


def lcg_statistics_init(algorithm_nb: int) -> NDArray:
    """
    Allocates the statistics array of the learning of LcgSolver: the global counters, then one per registered
    algorithm.

    :param algorithm_nb: the number of registered propagator algorithms
    :type algorithm_nb: int

    :return: the statistics array of the learning, zeroed
    :rtype: NDArray
    """
    return np.zeros(STATS_LCG_MAX + algorithm_nb, dtype=np.int64)


def lcg_statistics_as_dictionary(lcg_statistics: NDArray, algorithm_names: Sequence[str]) -> dict[str, int]:
    """
    Returns the statistics of the learning as a dictionary, labelled and with the per-algorithm tail broken out.

    The explanations of the propagators are given in total, then for each algorithm that the analysis asked at least
    once, with the name of the algorithm as a suffix. An algorithm with many explanations is the first candidate for an
    exact explain function (design/proposals/lcg.md, Part 5).

    :param lcg_statistics: the statistics array of the learning
    :type lcg_statistics: NDArray
    :param algorithm_names: the display name of each registered algorithm, indexed by algorithm
    :type algorithm_names: Sequence[str]

    :return: a dictionary mapping statistic labels to values
    :rtype: Dict[str, int]
    """
    dictionary = {
        STATS_LCG_LBL_BACKJUMP_LEVEL_NB: int(lcg_statistics[STATS_LCG_IDX_BACKJUMP_LEVEL_NB]),
        STATS_LCG_LBL_CLAUSE_EXPLANATION_NB: int(lcg_statistics[STATS_LCG_IDX_CLAUSE_EXPLANATION_NB]),
        STATS_LCG_LBL_CONFLICT_NB: int(lcg_statistics[STATS_LCG_IDX_CONFLICT_NB]),
        STATS_LCG_LBL_LEARNED_LBD_SUM: int(lcg_statistics[STATS_LCG_IDX_LEARNED_LBD_SUM]),
        STATS_LCG_LBL_LEARNED_LIT_NB: int(lcg_statistics[STATS_LCG_IDX_LEARNED_LIT_NB]),
        STATS_LCG_LBL_PROPAGATOR_EXPLANATION_NB: int(lcg_statistics[STATS_LCG_MAX:].sum()),
    }
    for algorithm, name in enumerate(algorithm_names):
        explanation_nb = int(lcg_statistics[STATS_LCG_MAX + algorithm])
        if explanation_nb:
            dictionary[f"{STATS_LCG_LBL_PROPAGATOR_EXPLANATION_NB}_{name}"] = explanation_nb
    return dictionary
