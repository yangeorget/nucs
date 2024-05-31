import numpy as np

from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.heuristics.smallest_domain_variable_heuristic import SmallestDomainVariableHeuristic
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

    def test_queens_8_solve_ff(self) -> None:
        problem = QueensProblem(8)
        solver = BacktrackSolver(problem, SmallestDomainVariableHeuristic(MinValueHeuristic()))
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 92


    def test_queens_9_solve_ff(self) -> None:
        problem = QueensProblem(9)
        solver = BacktrackSolver(problem, SmallestDomainVariableHeuristic(MinValueHeuristic()))
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 352
