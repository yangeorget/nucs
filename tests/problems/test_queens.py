from ncs.problems.queens_problem import QueensProblem
from ncs.solvers.backtrack_solver import BacktrackSolver


class TestQueens:
    def test_queens_8_problem(self) -> None:
        problem = QueensProblem(8)
        assert problem.domains.shape == (24, 2)
        assert problem.propagators[0].variables.shape == (8,)
        assert problem.propagators[1].variables.shape == (8,)
        assert problem.propagators[2].variables.shape == (8,)
        assert problem.propagators[3].variables.shape == (8, 2)
        assert problem.propagators[4].variables.shape == (8, 2)

    def test_queens_8_filter(self) -> None:
        problem = QueensProblem(8)
        assert problem.filter()

    def test_queens_8(self) -> None:
        problem = QueensProblem(8)
        solver = BacktrackSolver(problem)
        for _ in solver.solve():
            pass
        assert solver.statistics["solver.solutions.nb"] == 10000
        assert solver.statistics["backtracksolver.choicepoints.max"] == 2
