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
The steps of one node of the search, which the search loops share: the run_search_loop of BacktrackSolver
(backtrack/backtrack_solver.py) and the run_search_loop of LcgSolver (lcg/lcg_solver.py).

The steps are the search policy: when the search stops, which variable it branches on, and what it records after a
failure and after a decision. The two loops differ only in how they decide, propagate and backtrack.

Each step is always inlined, so that the loop that calls it compiles as one function.
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.numba_helper import VariableHeuristicFunctions
from nucs.solvers.interrupter import INTERRUPTION_DEADLINE, INTERRUPTION_EXTERNAL
from nucs.solvers.tree_search_solver import (
    SEARCH_CONTROL_CONFLICT_VARIABLE,
    SEARCH_CONTROL_DECISION_VARIABLE,
    SEARCH_CONTROL_FAILURE_NB,
    SEARCH_CONTROL_LAST_CONFLICT,
    SEARCH_CONTROL_RESTART_LIMIT,
    SOLVER_CHOICEPOINTS_FULL,
    SOLVER_INTERRUPTED,
    SOLVER_RESTART,
    SOLVER_RUNNING,
    SOLVER_UTRAIL_FULL,
)
from nucs.statistics import STATS_IDX_SOLVER_CHOICE_DEPTH, STATS_IDX_SOLVER_CHOICE_NB


@njit(cache=True, inline="always")
def stop_status(
    interruption: NDArray,
    search_control: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_headroom: int,
    choicepoint_top: NDArray,
    max_choicepoint: int,
) -> int:
    """
    Returns why the search must stop before the propagation of a node, or SOLVER_RUNNING when it can go on.

    The checks come before the state is touched: an interruption first, then the restart limit, so that the state the
    solver restarts from is not a half-done one, then the room of the arrays. The arrays belong to the caller, so the
    search stops for the solver to grow one, and does not overrun it: with boundscheck off, that is what would happen.

    :param interruption: the cells of the interrupter, see INTERRUPTION_*
    :type interruption: NDArray
    :param search_control: the control of the search, see SEARCH_CONTROL_*
    :type search_control: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_headroom: the undo trail entries that one node can add at most
    :type utrail_headroom: int
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
    :param max_choicepoint: the highest index of choice point at which a decision still fits in the stack
    :type max_choicepoint: int

    :return: SOLVER_INTERRUPTED, SOLVER_RESTART, SOLVER_UTRAIL_FULL, SOLVER_CHOICEPOINTS_FULL or SOLVER_RUNNING
    :rtype: int
    """
    if interruption[INTERRUPTION_EXTERNAL] | interruption[INTERRUPTION_DEADLINE]:
        return SOLVER_INTERRUPTED
    if 0 <= search_control[SEARCH_CONTROL_RESTART_LIMIT] <= search_control[SEARCH_CONTROL_FAILURE_NB]:
        return SOLVER_RESTART
    if utrail_top[0] + utrail_headroom > len(utrail_log):
        return SOLVER_UTRAIL_FULL
    if choicepoint_top[0] > max_choicepoint:
        return SOLVER_CHOICEPOINTS_FULL
    return SOLVER_RUNNING


@njit(cache=True, inline="always")
def select_variable(
    domains: NDArray,
    entailed: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_weights: NDArray,
    search_control: NDArray,
    variable_searches: NDArray,
    decision_variables: NDArray,
    decision_variables_offsets: NDArray,
    var_heuristic_fcts: VariableHeuristicFunctions,
    var_heuristic_params: NDArray,
    var_heuristic_params_offsets: NDArray,
    var_heuristic_params_shapes: NDArray,
) -> tuple[int, int]:
    """
    Returns the variable to branch on, and the search that owns it.

    The searches are sequential: the first search that still has an unbound decision variable owns the decision, and
    chooses the variable with its own variable heuristic. With last-conflict (Lecoutre et al. 2009), the variable of
    the last refuted decision, while it is unbound, takes the decision from the heuristic of the search that owns it.
    The searches before that search have no unbound variable left, so this keeps the order of the searches.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param entailed: the entailment flags of the propagators
    :type entailed: NDArray
    :param offsets: the offsets of the variables in the problem
    :type offsets: NDArray
    :param propagator_variables: the variables of the propagators
    :type propagator_variables: NDArray
    :param triggers: the propagators that the events on each variable trigger
    :type triggers: NDArray
    :param triggers_offsets: the offsets of the triggers
    :type triggers_offsets: NDArray
    :param propagator_weights: the weights of the propagators (see nucs.solvers.weights)
    :type propagator_weights: NDArray
    :param search_control: the control of the search, see SEARCH_CONTROL_*
    :type search_control: NDArray
    :param variable_searches: the search that owns each variable
    :type variable_searches: NDArray
    :param decision_variables: the decision variables of all the searches, one after the other
    :type decision_variables: NDArray
    :param decision_variables_offsets: the offsets of the decision variables of each search
    :type decision_variables_offsets: NDArray
    :param var_heuristic_fcts: the variable heuristic of each search
    :type var_heuristic_fcts: VariableHeuristicFunctions
    :param var_heuristic_params: the parameters of the variable heuristics, one after the other
    :type var_heuristic_params: NDArray
    :param var_heuristic_params_offsets: the offsets of the parameters of each variable heuristic
    :type var_heuristic_params_offsets: NDArray
    :param var_heuristic_params_shapes: the shape of the parameters of each variable heuristic
    :type var_heuristic_params_shapes: NDArray

    :return: the search that owns the decision, and the variable, -1 when no search has an unbound variable
    :rtype: Tuple[int, int]
    """
    # search_idx is set before the loop so that it has a value on every path
    search_idx = 0
    variable = -1
    conflict_variable = search_control[SEARCH_CONTROL_CONFLICT_VARIABLE]
    for search_idx in range(len(decision_variables_offsets) - 1):
        if (
            conflict_variable >= 0
            and variable_searches[conflict_variable] == search_idx
            and domains[conflict_variable, DOMAIN_MIN] < domains[conflict_variable, DOMAIN_MAX]
        ):
            variable = conflict_variable
        else:
            variable = var_heuristic_fcts[search_idx](
                decision_variables[decision_variables_offsets[search_idx] : decision_variables_offsets[search_idx + 1]],
                domains,
                entailed,
                offsets,
                propagator_variables,
                triggers,
                triggers_offsets,
                propagator_weights,
                var_heuristic_params[
                    var_heuristic_params_offsets[search_idx] : var_heuristic_params_offsets[search_idx + 1]
                ].reshape(var_heuristic_params_shapes[search_idx, 0], var_heuristic_params_shapes[search_idx, 1]),
            )
        if variable != -1:
            break
    return search_idx, variable


@njit(cache=True, inline="always")
def record_failure(search_control: NDArray) -> None:
    """
    Counts a failure, and makes the variable of the last decision the conflict variable of last-conflict.

    A failure that follows a refutation keeps the conflict variable: after x fails on all its values, the search
    refutes an earlier decision on y, and x must then be tried first, to test y.

    :param search_control: the control of the search, see SEARCH_CONTROL_*
    :type search_control: NDArray
    """
    search_control[SEARCH_CONTROL_FAILURE_NB] += 1
    if search_control[SEARCH_CONTROL_DECISION_VARIABLE] >= 0:
        search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = search_control[SEARCH_CONTROL_DECISION_VARIABLE]
        search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1


@njit(cache=True, inline="always")
def record_decision(statistics: NDArray, search_control: NDArray, variable: int, choicepoint: int) -> None:
    """
    Counts a decision, keeps its variable for last-conflict, and keeps the depth of the search.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param search_control: the control of the search, see SEARCH_CONTROL_*
    :type search_control: NDArray
    :param variable: the variable of the decision
    :type variable: int
    :param choicepoint: the index of the top of the choice points after the decision
    :type choicepoint: int
    """
    statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
    if search_control[SEARCH_CONTROL_LAST_CONFLICT]:
        search_control[SEARCH_CONTROL_DECISION_VARIABLE] = variable
    statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = max(statistics[STATS_IDX_SOLVER_CHOICE_DEPTH], choicepoint)
