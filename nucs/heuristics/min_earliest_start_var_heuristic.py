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
import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN


@njit(cache=True)
def min_earliest_start_var_heuristic(
    decision_variables: NDArray,
    domains: NDArray,
    entailed: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_weights: NDArray,
    params: NDArray,
) -> int:
    """
    Chooses the unbound task with the smallest earliest start time, the selection rule of the Set Times search.

    Each decision variable is the start time of a task, so its domain minimum is the task's earliest start and
    its maximum the latest start. This heuristic returns the unbound task that can start soonest, ties broken on
    the smallest latest start (the most urgent). Paired with the ``min_value`` domain heuristic -- whose two
    branches "bind the start to its earliest value" and "forbid that value" are exactly Set Times' schedule and
    postpone decisions -- it realizes the Set Times scheme: repeatedly commit the soonest-startable task to its
    earliest time, or postpone it and let the disjunctive propagator push its earliest start to the next
    feasible point.

    :param decision_variables: the decision variables, the start times of the tasks
    :type decision_variables: NDArray
    :param domains: the domains
    :type domains: NDArray
    :param entailed: whether each propagator is entailed, unused here
    :type entailed: NDArray
    :param offsets: the offsets of the propagators, unused here
    :type offsets: NDArray
    :param propagator_variables: the variables of the propagators, unused here
    :type propagator_variables: NDArray
    :param triggers: the propagators to wake, grouped by variable and event, unused here
    :type triggers: NDArray
    :param triggers_offsets: the offsets of each (variable, event) slice of triggers, unused here
    :type triggers_offsets: NDArray
    :param propagator_weights: the failure weights of the propagators, unused here
    :type propagator_weights: NDArray
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the variable, or -1 when every task is bound
    :rtype: int
    """
    best_variable = -1
    best_earliest_start = sys.maxsize
    best_latest_start = sys.maxsize
    for variable in decision_variables:
        earliest_start = domains[variable, DOMAIN_MIN]
        latest_start = domains[variable, DOMAIN_MAX]
        # unbound, and lexicographically better on (earliest_start, latest_start)
        if earliest_start < latest_start and (
            earliest_start < best_earliest_start
            or (earliest_start == best_earliest_start and latest_start < best_latest_start)
        ):
            best_earliest_start = earliest_start
            best_latest_start = latest_start
            best_variable = variable
    return best_variable
