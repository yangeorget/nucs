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

from nucs.constants import MAX, MIN


@njit(cache=True, fastmath=True)
def min_earliest_start_var_heuristic(
    decision_variables: NDArray, domains_stk: NDArray, top: int, params: NDArray
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
    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the variable, or -1 when every task is bound
    :rtype: int
    """
    best_variable = -1
    best_earliest_start = sys.maxsize
    best_latest_start = sys.maxsize
    for variable in decision_variables:
        earliest_start = domains_stk[top, variable, MIN]
        latest_start = domains_stk[top, variable, MAX]
        if earliest_start < latest_start:  # unbound
            if earliest_start < best_earliest_start or (
                earliest_start == best_earliest_start and latest_start < best_latest_start
            ):
                best_earliest_start = earliest_start
                best_latest_start = latest_start
                best_variable = variable
    return best_variable
