from typing import Dict

import numpy as np
from numpy.typing import NDArray


def init_statistics() -> NDArray:
    """
    Inits a Numpy array for storing the statistics.
    :return: a Numpy array
    """
    return np.array([0] * STATS_MAX, dtype=np.int64)


def get_statistics(stats: NDArray) -> Dict[str, int]:
    """
    Returns the statistics as a dictionary.
    :param stats: a Numpy array of statistics
    :return: a dictionary
    """
    return {
        "OPTIMIZER_SOLUTION_NB": int(stats[STATS_OPTIMIZER_SOLUTION_NB]),
        "PROBLEM_FILTER_NB": int(stats[STATS_PROBLEM_FILTER_NB]),
        "PROBLEM_PROPAGATOR_NB": int(stats[STATS_PROBLEM_PROPAGATOR_NB]),
        "PROBLEM_VARIABLE_NB": int(stats[STATS_PROBLEM_VARIABLE_NB]),
        "PROPAGATOR_ENTAILMENT_NB": int(stats[STATS_PROPAGATOR_ENTAILMENT_NB]),
        "PROPAGATOR_FILTER_NB": int(stats[STATS_PROPAGATOR_FILTER_NB]),
        "PROPAGATOR_FILTER_NO_CHANGE_NB": int(stats[STATS_PROPAGATOR_FILTER_NO_CHANGE_NB]),
        "PROPAGATOR_INCONSISTENCY_NB": int(stats[STATS_PROPAGATOR_INCONSISTENCY_NB]),
        "SOLVER_BACKTRACK_NB": int(stats[STATS_SOLVER_BACKTRACK_NB]),
        "SOLVER_CHOICE_NB": int(stats[STATS_SOLVER_CHOICE_NB]),
        "SOLVER_CHOICE_DEPTH": int(stats[STATS_SOLVER_CHOICE_DEPTH]),
        "SOLVER_SOLUTION_NB": int(stats[STATS_SOLVER_SOLUTION_NB]),
    }


STATS_MAX = 12
(
    STATS_OPTIMIZER_SOLUTION_NB,
    STATS_PROBLEM_FILTER_NB,
    STATS_PROBLEM_PROPAGATOR_NB,
    STATS_PROBLEM_VARIABLE_NB,
    STATS_PROPAGATOR_ENTAILMENT_NB,
    STATS_PROPAGATOR_FILTER_NB,
    STATS_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_PROPAGATOR_INCONSISTENCY_NB,
    STATS_SOLVER_BACKTRACK_NB,
    STATS_SOLVER_CHOICE_NB,
    STATS_SOLVER_CHOICE_DEPTH,
    STATS_SOLVER_SOLUTION_NB,
) = tuple(range(STATS_MAX))
