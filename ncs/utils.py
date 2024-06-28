import numpy as np
from numpy.typing import NDArray

MIN = 0
MAX = 1

STATS_MAX = 6
STATS_PROBLEM_FILTERS_NB = 0
STATS_PROBLEM_PROPAGATORS_FILTERS_NB = 1
STATS_SOLVER_BACKTRACKS_NB = 2
STATS_SOLVER_CHOICES_NB = 3
STATS_SOLVER_CP_MAX = 4
STATS_SOLVER_SOLUTIONS_NB = 5


def stats_inc(stats: NDArray, key: int) -> None:
    stats[key] += 1


def stats_max(stats: NDArray, key: int, value: int) -> None:
    stats[key] = max(stats[key], value)


def stats_init() -> NDArray:
    return np.array([0] * STATS_MAX, dtype=np.int32)


def stats_print(stats: NDArray) -> None:
    print(
        {
            "PROBLEM_FILTERS_NB": int(stats[STATS_PROBLEM_FILTERS_NB]),
            "PROBLEM_PROPAGATORS_FILTERS_NB": int(stats[STATS_PROBLEM_PROPAGATORS_FILTERS_NB]),
            "SOLVER_BACKTRACKS_NB": int(stats[STATS_SOLVER_BACKTRACKS_NB]),
            "SOLVER_CHOICES_NB": int(stats[STATS_SOLVER_CHOICES_NB]),
            "SOLVER_CP_MAX": int(stats[STATS_SOLVER_CP_MAX]),
            "SOLVER_SOLUTIONS_NB": int(stats[STATS_SOLVER_SOLUTIONS_NB]),
        }
    )
