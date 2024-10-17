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
from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray


def init_statistics() -> NDArray:
    """
    Inits a Numpy array for storing the statistics.
    :return: a Numpy array
    """
    return np.array([0] * STATS_MAX, dtype=np.int64)


def sum_stats(stats: List[NDArray], index: int) -> int:
    return sum(int(s[index]) for s in stats)


def max_stats(stats: List[NDArray], index: int) -> int:
    return max(int(s[index]) for s in stats)


def get_statistics(stats: Union[NDArray, List[NDArray]]) -> Dict[str, int]:
    """
    Returns the statistics as a dictionary.
    :param stats: a Numpy array of statistics
    :return: a dictionary
    """
    if isinstance(stats, list):
        return {
            STATS_LBL_OPTIMIZER_SOLUTION_NB: sum_stats(stats, STATS_IDX_OPTIMIZER_SOLUTION_NB),
            STATS_LBL_PROBLEM_FILTER_NB: sum_stats(stats, STATS_IDX_PROBLEM_FILTER_NB),
            STATS_LBL_PROBLEM_PROPAGATOR_NB: int(stats[0][STATS_IDX_PROBLEM_PROPAGATOR_NB]),
            STATS_LBL_PROBLEM_VARIABLE_NB: int(stats[0][STATS_IDX_PROBLEM_VARIABLE_NB]),
            STATS_LBL_PROPAGATOR_ENTAILMENT_NB: sum_stats(stats, STATS_IDX_PROPAGATOR_ENTAILMENT_NB),
            STATS_LBL_PROPAGATOR_FILTER_NB: sum_stats(stats, STATS_IDX_PROPAGATOR_FILTER_NB),
            STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB: sum_stats(stats, STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB),
            STATS_LBL_PROPAGATOR_INCONSISTENCY_NB: sum_stats(stats, STATS_IDX_PROPAGATOR_INCONSISTENCY_NB),
            STATS_LBL_SOLVER_BACKTRACK_NB: sum_stats(stats, STATS_IDX_SOLVER_BACKTRACK_NB),
            STATS_LBL_SOLVER_CHOICE_NB: sum_stats(stats, STATS_IDX_SOLVER_CHOICE_NB),
            STATS_LBL_SOLVER_CHOICE_DEPTH: max_stats(stats, STATS_IDX_SOLVER_CHOICE_DEPTH),
            STATS_LBL_SOLVER_SOLUTION_NB: sum_stats(stats, STATS_IDX_SOLVER_SOLUTION_NB),
        }
    else:
        return {
            STATS_LBL_OPTIMIZER_SOLUTION_NB: int(stats[STATS_IDX_OPTIMIZER_SOLUTION_NB]),
            STATS_LBL_PROBLEM_FILTER_NB: int(stats[STATS_IDX_PROBLEM_FILTER_NB]),
            STATS_LBL_PROBLEM_PROPAGATOR_NB: int(stats[STATS_IDX_PROBLEM_PROPAGATOR_NB]),
            STATS_LBL_PROBLEM_VARIABLE_NB: int(stats[STATS_IDX_PROBLEM_VARIABLE_NB]),
            STATS_LBL_PROPAGATOR_ENTAILMENT_NB: int(stats[STATS_IDX_PROPAGATOR_ENTAILMENT_NB]),
            STATS_LBL_PROPAGATOR_FILTER_NB: int(stats[STATS_IDX_PROPAGATOR_FILTER_NB]),
            STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB: int(stats[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB]),
            STATS_LBL_PROPAGATOR_INCONSISTENCY_NB: int(stats[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB]),
            STATS_LBL_SOLVER_BACKTRACK_NB: int(stats[STATS_IDX_SOLVER_BACKTRACK_NB]),
            STATS_LBL_SOLVER_CHOICE_NB: int(stats[STATS_IDX_SOLVER_CHOICE_NB]),
            STATS_LBL_SOLVER_CHOICE_DEPTH: int(stats[STATS_IDX_SOLVER_CHOICE_DEPTH]),
            STATS_LBL_SOLVER_SOLUTION_NB: int(stats[STATS_IDX_SOLVER_SOLUTION_NB]),
        }


STATS_MAX = 12
(
    STATS_IDX_OPTIMIZER_SOLUTION_NB,
    STATS_IDX_PROBLEM_FILTER_NB,
    STATS_IDX_PROBLEM_PROPAGATOR_NB,
    STATS_IDX_PROBLEM_VARIABLE_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_SOLUTION_NB,
) = tuple(range(STATS_MAX))

STATS_LBL_OPTIMIZER_SOLUTION_NB = "OPTIMIZER_SOLUTION_NB"
STATS_LBL_PROBLEM_FILTER_NB = "PROBLEM_FILTER_NB"
STATS_LBL_PROBLEM_PROPAGATOR_NB = "PROBLEM_PROPAGATOR_NB"
STATS_LBL_PROBLEM_VARIABLE_NB = "PROBLEM_VARIABLE_NB"
STATS_LBL_PROPAGATOR_ENTAILMENT_NB = "PROPAGATOR_ENTAILMENT_NB"
STATS_LBL_PROPAGATOR_FILTER_NB = "PROPAGATOR_FILTER_NB"
STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB = "PROPAGATOR_FILTER_NO_CHANGE_NB"
STATS_LBL_PROPAGATOR_INCONSISTENCY_NB = "PROPAGATOR_INCONSISTENCY_NB"
STATS_LBL_SOLVER_BACKTRACK_NB = "SOLVER_BACKTRACK_NB"
STATS_LBL_SOLVER_CHOICE_NB = "SOLVER_CHOICE_NB"
STATS_LBL_SOLVER_CHOICE_DEPTH = "SOLVER_CHOICE_DEPTH"
STATS_LBL_SOLVER_SOLUTION_NB = "SOLVER_SOLUTION_NB"
