import numpy as np
from numpy.typing import NDArray

START = 0
END = 1

MIN = 0
MAX = 1

STATS_MAX = 8
STATS_OPTIMIZER_SOLUTION_NB = 0
STATS_PROBLEM_FILTER_NB = 1
STATS_PROPAGATOR_FILTER_NB = 2
STATS_PROPAGATOR_FILTER_NO_CHANGE = 3
STATS_SOLVER_BACKTRACK_NB = 4
STATS_SOLVER_CHOICE_NB = 5
STATS_SOLVER_CHOICE_DEPTH = 6
STATS_SOLVER_SOLUTION_NB = 7


def statistics_init() -> NDArray:
    """
    Inits a Numpy array for storing the statistics.
    :return: a Numpy array
    """
    return np.array([0] * STATS_MAX, dtype=np.int32)


def statistics_print(stats: NDArray) -> None:
    """
    Pretty-prints an array of statistics.
    :param stats: a Numpy array of statistics
    """
    print(
        {
            "PROBLEM_FILTERS_NB": int(stats[STATS_PROBLEM_FILTER_NB]),
            "PROBLEM_PROPAGATORS_FILTERS_NB": int(stats[STATS_PROPAGATOR_FILTER_NB]),
            "PROBLEM_PROPAGATORS_FILTERS_NO_CHANGE": int(stats[STATS_PROPAGATOR_FILTER_NO_CHANGE]),
            "SOLVER_BACKTRACKS_NB": int(stats[STATS_SOLVER_BACKTRACK_NB]),
            "SOLVER_CHOICES_NB": int(stats[STATS_SOLVER_CHOICE_NB]),
            "SOLVER_CP_MAX": int(stats[STATS_SOLVER_CHOICE_DEPTH]),
            "SOLVER_SOLUTIONS_NB": int(stats[STATS_SOLVER_SOLUTION_NB]),
        }
    )
