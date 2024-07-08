from ncs.heuristics.variable_heuristic import (
    VariableHeuristic,
    first_not_instantiated_variable_heuristic,
    min_value_domain_heuristic,
    smallest_domain_variable_heuristic,
)
from ncs.problems.queens.queens_problem import QueensProblem
from ncs.solvers.backtrack_solver import BacktrackSolver
from ncs.utils import STATS_SOLVER_SOLUTIONS_NB, statistics_print


class TestQueens:
    def test_queens_8_filter(self) -> None:
        problem = QueensProblem(8)
        assert problem.filter()

    def test_queens_1_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(1))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 1

    def test_queens_2_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(2))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 0

    def test_queens_3_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(3))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 0

    def test_queens_4_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(4))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 2

    def test_queens_5_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(5))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 10

    def test_queens_6_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(6))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 4

    def test_queens_8_solve(self) -> None:
        solver = BacktrackSolver(QueensProblem(8))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 92

    def test_queens_8_solve_ff(self) -> None:
        solver = BacktrackSolver(
            QueensProblem(8), VariableHeuristic(smallest_domain_variable_heuristic, min_value_domain_heuristic)
        )
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 92

    def test_queens_9_solve_ff(self) -> None:
        solver = BacktrackSolver(
            QueensProblem(9), VariableHeuristic(smallest_domain_variable_heuristic, min_value_domain_heuristic)
        )
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 352


if __name__ == "__main__":
    solver = BacktrackSolver(
        QueensProblem(12), VariableHeuristic(first_not_instantiated_variable_heuristic, min_value_domain_heuristic)
    )
    solver.solve_all()
    statistics_print(solver.statistics)
