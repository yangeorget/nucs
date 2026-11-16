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
import numpy as np
import pytest

from nucs.problems.problem import Problem
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.interrupter import INTERRUPTION_DEADLINE, INTERRUPTION_EXTERNAL, INTERRUPTION_WIDTH
from nucs.solvers.search_steps import record_decision, record_failure, select_variable, stop_status
from nucs.solvers.searches import Search
from nucs.solvers.tree_search_solver import (
    SEARCH_CONTROL_CONFLICT_VARIABLE,
    SEARCH_CONTROL_DECISION_VARIABLE,
    SEARCH_CONTROL_FAILURE_NB,
    SEARCH_CONTROL_LAST_CONFLICT,
    SEARCH_CONTROL_RESTART_LIMIT,
    SEARCH_CONTROL_WIDTH,
    SOLVER_CHOICEPOINTS_FULL,
    SOLVER_INTERRUPTED,
    SOLVER_RESTART,
    SOLVER_RUNNING,
    SOLVER_UTRAIL_FULL,
)
from nucs.statistics import STATS_IDX_SOLVER_CHOICE_DEPTH, STATS_IDX_SOLVER_CHOICE_NB, statistics_init


class TestSearchSteps:
    @pytest.mark.parametrize(
        "interruption_idx,restart_limit,utrail_sz,choicepoint,expected",
        [
            (-1, -1, 0, 0, SOLVER_RUNNING),  # nothing stops the search
            (INTERRUPTION_EXTERNAL, 0, 8, 9, SOLVER_INTERRUPTED),  # interrupt() comes before all the other checks
            (INTERRUPTION_DEADLINE, -1, 0, 0, SOLVER_INTERRUPTED),  # a timeout
            (-1, 3, 8, 9, SOLVER_RESTART),  # 3 failures reach the limit, before the arrays are checked
            (-1, 4, 0, 0, SOLVER_RUNNING),  # 3 failures are below the limit
            (-1, -1, 7, 9, SOLVER_UTRAIL_FULL),  # 7 entries and a headroom of 2 do not fit in 8
            (-1, -1, 6, 9, SOLVER_CHOICEPOINTS_FULL),  # 6 entries fit; choice point 9 is above the max of 8
            (-1, -1, 6, 8, SOLVER_RUNNING),  # choice point 8 is the max
        ],
    )
    def test_stop_status(
        self, interruption_idx: int, restart_limit: int, utrail_sz: int, choicepoint: int, expected: int
    ) -> None:
        interruption = np.zeros(INTERRUPTION_WIDTH, dtype=np.bool_)
        if interruption_idx >= 0:
            interruption[interruption_idx] = True
        search_control = np.full(SEARCH_CONTROL_WIDTH, -1, dtype=np.int64)
        search_control[SEARCH_CONTROL_RESTART_LIMIT] = restart_limit
        search_control[SEARCH_CONTROL_FAILURE_NB] = 3
        utrail_log = np.zeros((8, 2), dtype=np.int32)
        utrail_top = np.array([utrail_sz], dtype=np.int32)
        choicepoint_top = np.array([choicepoint], dtype=np.int32)
        assert stop_status(interruption, search_control, utrail_log, utrail_top, 2, choicepoint_top, 8) == expected

    @pytest.mark.parametrize(
        "bound_variables,conflict_variable,expected",
        [
            ([], -1, (0, 0)),  # the first search owns the decision
            ([0, 1], -1, (1, 2)),  # the first search has no unbound variable left: the second one owns the decision
            ([], 1, (0, 1)),  # last-conflict: the unbound conflict variable takes the decision of its search
            ([1], 1, (0, 0)),  # the conflict variable is bound: the heuristic decides
            ([], 3, (0, 0)),  # the conflict variable belongs to the second search: the order of the searches stays
            ([0, 1], 3, (1, 3)),  # the second search owns the decision, and its conflict variable takes it
            ([0, 1, 2, 3], -1, (1, -1)),  # no unbound variable: the last search, and -1
        ],
    )
    def test_select_variable(
        self, bound_variables: list[int], conflict_variable: int, expected: tuple[int, int]
    ) -> None:
        # two searches, on x0, x1 and on x2, x3, with first_not_instantiated
        solver = BacktrackSolver(Problem([(0, 3)] * 4), searches=[Search([0, 1]), Search([2, 3])])
        for variable in bound_variables:
            solver.domains[variable] = [2, 2]
        solver.search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = conflict_variable
        flat = solver.flat_searches
        result = select_variable(
            solver.domains,
            solver.entailed,
            solver.problem.offsets,
            solver.problem.propagator_variables,
            solver.problem.triggers,
            solver.problem.triggers_offsets,
            solver.propagator_weights,
            solver.search_control,
            flat.variable_searches,
            flat.decision_variables,
            flat.decision_variables_offsets,
            solver.var_heuristic_fcts,
            flat.var_heuristic_params,
            flat.var_heuristic_params_offsets,
            flat.var_heuristic_params_shapes,
        )
        assert result == expected

    @pytest.mark.parametrize(
        "decision_variable,expected_conflict_variable",
        [
            (2, 2),  # the variable of the last decision becomes the conflict variable
            (-1, 5),  # a failure after a refutation keeps the conflict variable
        ],
    )
    def test_record_failure(self, decision_variable: int, expected_conflict_variable: int) -> None:
        search_control = np.full(SEARCH_CONTROL_WIDTH, -1, dtype=np.int64)
        search_control[SEARCH_CONTROL_FAILURE_NB] = 3
        search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = 5
        search_control[SEARCH_CONTROL_DECISION_VARIABLE] = decision_variable
        record_failure(search_control)
        assert search_control[SEARCH_CONTROL_FAILURE_NB] == 4
        assert search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] == expected_conflict_variable
        assert search_control[SEARCH_CONTROL_DECISION_VARIABLE] == -1

    @pytest.mark.parametrize(
        "last_conflict,choicepoint,expected_decision_variable,expected_depth",
        [
            (1, 7, 2, 7),  # last-conflict keeps the variable of the decision; a deeper choice point is the new depth
            (0, 3, -1, 5),  # without last-conflict, no variable is kept; the depth stays at its max
        ],
    )
    def test_record_decision(
        self, last_conflict: int, choicepoint: int, expected_decision_variable: int, expected_depth: int
    ) -> None:
        statistics = statistics_init(1)
        statistics[STATS_IDX_SOLVER_CHOICE_NB] = 10
        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = 5
        search_control = np.full(SEARCH_CONTROL_WIDTH, -1, dtype=np.int64)
        search_control[SEARCH_CONTROL_LAST_CONFLICT] = last_conflict
        record_decision(statistics, search_control, 2, choicepoint)
        assert statistics[STATS_IDX_SOLVER_CHOICE_NB] == 11
        assert search_control[SEARCH_CONTROL_DECISION_VARIABLE] == expected_decision_variable
        assert statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] == expected_depth
