import numpy as np

from ncs.problems.queens_problem import QueensProblem
from ncs.solvers.backtrack_solver import BacktrackSolver


class TestQueens:
    def test_queens_8_problem(self) -> None:
        problem = QueensProblem(8)
        assert problem.domains.shape == (24, 2)
        assert problem.propagators[0].variables.shape == (8,)
        assert problem.propagators[1].variables.shape == (8,)
        assert problem.propagators[2].variables.shape == (8,)
        assert problem.propagators[3].variables.shape == (16,)
        assert problem.propagators[4].variables.shape == (16,)

    def test_queens_8_filter(self) -> None:
        problem = QueensProblem(8)
        assert problem.filter()

    def test_queens_4_solve(self) -> None:
        problem = QueensProblem(4)
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 2

    def test_queens_5_solve_when_q0_is_2(self) -> None:
        problem = QueensProblem(5)
        problem.domains[0] = np.array([2, 2])
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 2

    def test_queens_5_solve(self) -> None:
        problem = QueensProblem(5)
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 10

    def test_queens_8_solve(self) -> None:
        problem = QueensProblem(8)
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 92
