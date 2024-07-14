import numpy as np
from numpy.typing import NDArray

START = 0
END = 1

MIN = 0
MAX = 1

STATS_MAX = 7
STATS_OPTIMIZER_SOLUTION_NB = 0
STATS_PROBLEM_FILTER_NB = 1
STATS_PROPAGATOR_FILTER_NB = 2
STATS_SOLVER_BACKTRACK_NB = 3
STATS_SOLVER_CHOICE_NB = 4
STATS_SOLVER_CHOICE_DEPTH = 5
STATS_SOLVER_SOLUTION_NB = 6


def statistics_init() -> NDArray:
    return np.array([0] * STATS_MAX, dtype=np.int32)


def statistics_print(stats: NDArray) -> None:
    print(
        {
            "PROBLEM_FILTERS_NB": int(stats[STATS_PROBLEM_FILTER_NB]),
            "PROBLEM_PROPAGATORS_FILTERS_NB": int(stats[STATS_PROPAGATOR_FILTER_NB]),
            "SOLVER_BACKTRACKS_NB": int(stats[STATS_SOLVER_BACKTRACK_NB]),
            "SOLVER_CHOICES_NB": int(stats[STATS_SOLVER_CHOICE_NB]),
            "SOLVER_CP_MAX": int(stats[STATS_SOLVER_CHOICE_DEPTH]),
            "SOLVER_SOLUTIONS_NB": int(stats[STATS_SOLVER_SOLUTION_NB]),
        }
    )
