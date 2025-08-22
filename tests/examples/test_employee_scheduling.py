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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from nucs.examples.employee_scheduling.employee_scheduling_problem import EmployeeSchedulingProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE
from nucs.solvers.backtrack_solver import BacktrackSolver


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

    def test_employee_scheduling(self) -> None:
        problem = EmployeeSchedulingProblem()
        solver = BacktrackSolver(
            problem, decision_variables=problem.requested_shifts, dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE
        )
        solution = solver.maximize(problem.satisfied_request_nb)
        assert solution is not None
        assert solution[problem.satisfied_request_nb] == 13
