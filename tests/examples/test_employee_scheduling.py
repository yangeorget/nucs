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
from collections.abc import Callable

import pytest

from nucs.constants import DOMAIN_MAX
from nucs.examples.employee_scheduling.employee_scheduling_problem import EmployeeSchedulingProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search
from nucs.statistics import STATS_IDX_SOLVER_CHOICE_NB


class TestEmployeeScheduling:
    def test_nurses(self) -> None:
        assert list(EmployeeSchedulingProblem().nurses(0, 0)) == [0, 1, 2, 3, 4]
        assert list(EmployeeSchedulingProblem().nurses(0, 1)) == [5, 6, 7, 8, 9]
        assert list(EmployeeSchedulingProblem().nurses(1, 0)) == [15, 16, 17, 18, 19]
        assert list(EmployeeSchedulingProblem().nurses(1, 1)) == [20, 21, 22, 23, 24]

    def test_shifts(self) -> None:
        assert list(EmployeeSchedulingProblem().shifts(0, 0)) == [0, 5, 10]
        assert list(EmployeeSchedulingProblem().shifts(0, 1)) == [1, 6, 11]
        assert list(EmployeeSchedulingProblem().shifts(1, 0)) == [15, 20, 25]
        assert list(EmployeeSchedulingProblem().shifts(1, 1)) == [16, 21, 26]

    @pytest.mark.parametrize("last_conflict, choice_nb", [(False, 1245), (True, 405)])
    def test_find_best(self, last_conflict: bool, choice_nb: int) -> None:
        """Last-conflict proves the same optimum with 405 choices instead of 1,245."""
        problem = EmployeeSchedulingProblem()
        solver = BacktrackSolver(
            problem,
            searches=[Search(decision_variables=problem.requested_shifts, dom_heuristic=DOM_HEURISTIC_MAX_VALUE)],
            last_conflict=last_conflict,
        )
        solution = solver.find_best(problem.satisfied_request_nb, bound=DOMAIN_MAX)
        assert solution is not None
        assert solution[problem.satisfied_request_nb] == 13
        assert solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] == choice_nb

    def test_main(self, run_example_main: Callable[..., BacktrackSolver]) -> None:
        """The example uses last-conflict."""
        solver = run_example_main("nucs.examples.employee_scheduling")
        assert solver.statistics[STATS_IDX_SOLVER_CHOICE_NB] == 405
