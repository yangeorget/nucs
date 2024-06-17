from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.heuristics.smallest_domain_variable_heuristic import (
    SmallestDomainVariableHeuristic,
)
from ncs.problems.queens_problem import QueensProblem
from ncs.solvers.backtrack_solver import BacktrackSolver


class TestQueens:
    def test_queens_8_filter(self) -> None:
        problem = QueensProblem(8)
        assert problem.filter()

    def test_queens_1_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(1))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 1

    def test_queens_2_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(2))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 0

    def test_queens_3_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(3))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 0

    def test_queens_4_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(4))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 2

    def test_queens_5_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(5))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 10

    def test_queens_6_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(6))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 4

    def test_queens_8_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(8))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 92

    def test_queens_8_solve_ff(self) -> None:
        solver = BacktrackSolver(QueensProblem(8), SmallestDomainVariableHeuristic(MinValueHeuristic()))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 92

    def test_queens_9_solve_ff(self) -> None:
        solver = BacktrackSolver(QueensProblem(9), SmallestDomainVariableHeuristic(MinValueHeuristic()))
        solver.solve_all()
        assert solver.statistics["solver.solutions.nb"] == 352


if __name__ == "__main__":
    solver = BacktrackSolver(QueensProblem(10), SmallestDomainVariableHeuristic(MinValueHeuristic()))
    solver.solve_all()
    print(solver.statistics)
