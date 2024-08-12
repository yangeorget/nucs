import numpy as np
from numpy.typing import NDArray


def statistics_init() -> NDArray:
    """
    Inits a Numpy array for storing the statistics.
    :return: a Numpy array
    """
    return np.array([0] * STATS_MAX, dtype=np.int32, order="C")


def statistics_print(stats: NDArray) -> None:
    """
    Pretty-prints an array of statistics.
    :param stats: a Numpy array of statistics
    """
    print(
        {
            "OPTIMIZER_SOLUTION_NB": int(stats[STATS_OPTIMIZER_SOLUTION_NB]),
            "PROBLEM_FILTERS_NB": int(stats[STATS_PROBLEM_FILTER_NB]),
            "PROPAGATOR_FILTERS_NB": int(stats[STATS_PROPAGATOR_FILTER_NB]),
            "PROPAGATOR_FILTERS_NO_CHANGE_NB": int(stats[STATS_PROPAGATOR_FILTER_NO_CHANGE_NB]),
            "PROPAGATOR_INCONSISTENCY_NB": int(stats[STATS_PROPAGATOR_INCONSISTENCY_NB]),
            "SOLVER_BACKTRACK_NB": int(stats[STATS_SOLVER_BACKTRACK_NB]),
            "SOLVER_CHOICE_NB": int(stats[STATS_SOLVER_CHOICE_NB]),
            "SOLVER_CHOICE_DEPTH": int(stats[STATS_SOLVER_CHOICE_DEPTH]),
            "SOLVER_SOLUTION_NB": int(stats[STATS_SOLVER_SOLUTION_NB]),
        }
    )


STATS_MAX = 9
STATS_OPTIMIZER_SOLUTION_NB = 0
STATS_PROBLEM_FILTER_NB = 1
STATS_PROPAGATOR_FILTER_NB = 2
STATS_PROPAGATOR_FILTER_NO_CHANGE_NB = 3
STATS_PROPAGATOR_INCONSISTENCY_NB = 4
STATS_SOLVER_BACKTRACK_NB = 5
STATS_SOLVER_CHOICE_NB = 6
STATS_SOLVER_CHOICE_DEPTH = 7
STATS_SOLVER_SOLUTION_NB = 8
