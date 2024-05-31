import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent_puget_n3 import AlldifferentPugetN3
from ncs.propagators.sum import Sum
from ncs.solvers.backtrack_solver import BacktrackSolver


class TestBacktrackSolver:
    def test_solve_and_count(self) -> None:
        domains = np.array([[0, 99], [0, 99]])
        problem = Problem(domains)
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 10000
        assert solver.statistics["backtracksolver.choicepoints.max"] == 2

    def test_solve(self) -> None:
        domains = np.array([[0, 1], [0, 1]])
        problem = Problem(domains)
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 4
        assert np.all(solutions[0] == np.array([[0, 0], [0, 0]]))
        assert np.all(solutions[1] == np.array([[0, 0], [1, 1]]))
        assert np.all(solutions[2] == np.array([[1, 1], [0, 0]]))
        assert np.all(solutions[3] == np.array([[1, 1], [1, 1]]))
        assert solver.statistics["solver.solutions.nb"] == 4
        assert solver.statistics["backtracksolver.choicepoints.max"] == 2

    def test_solve_sum_1(self) -> None:
        domains = np.array([[0, 2], [0, 2], [4, 6]])
        problem = Problem(domains, [Sum([2, 0, 1])])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert np.all(solutions == np.array([[2, 2], [2, 2], [4, 4]]))
        assert solver.statistics["solver.solutions.nb"] == 1
        assert solver.statistics["problem.filters.nb"] == 1
        assert solver.statistics["backtracksolver.choicepoints.max"] == 0
        assert solver.statistics["backtracksolver.backtracks.nb"] == 0

    def test_solve_sum_3(self) -> None:
        domains = np.array([[0, 1], [0, 1], [0, 1]])
        problem = Problem(domains, [Sum([2, 0, 1])])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 3
        assert np.all(solutions[0] == np.array([[0, 0], [0, 0], [0, 0]]))
        assert np.all(solutions[1] == np.array([[0, 0], [1, 1], [1, 1]]))
        assert np.all(solutions[2] == np.array([[1, 1], [0, 0], [1, 1]]))
        assert solver.statistics["solver.solutions.nb"] == 3
        assert solver.statistics["backtracksolver.choicepoints.max"] == 2

    def test_solve_sum_ko(self) -> None:
        domains = np.array([[1, 2], [1, 2], [0, 1]])
        problem = Problem(domains, [Sum([2, 0, 1])])
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 0
        assert solver.statistics["problem.filters.nb"] == 1
        assert solver.statistics["backtracksolver.choicepoints.max"] == 0

    def test_solve_alldifferent(self) -> None:
        domains = np.array([[0, 2], [0, 2], [0, 2]])
        problem = Problem(domains, [AlldifferentPugetN3([0, 1, 2])])
        solver = BacktrackSolver(problem)
        solutions = [solution for solution in solver.solve()]
        assert len(solutions) == 6
        assert np.all(solutions[0] == np.array([[0, 0], [1, 1], [2, 2]]))
        assert np.all(solutions[1] == np.array([[0, 0], [2, 2], [1, 1]]))
        assert np.all(solutions[2] == np.array([[1, 1], [0, 0], [2, 2]]))
        assert np.all(solutions[3] == np.array([[1, 1], [2, 2], [0, 0]]))
        assert np.all(solutions[4] == np.array([[2, 2], [0, 0], [1, 1]]))
        assert np.all(solutions[5] == np.array([[2, 2], [1, 1], [0, 0]]))
