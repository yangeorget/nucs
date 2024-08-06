import argparse

import pytest

from ncs.heuristics.variable_heuristic import (
    VariableHeuristic,
    first_not_instantiated_var_heuristic,
    min_value_dom_heuristic,
    smallest_domain_var_heuristic,
)
from ncs.problems.queens_problem import QueensProblem
from ncs.solvers.backtrack_solver import BacktrackSolver
from ncs.utils import STATS_SOLVER_SOLUTION_NB, stats_print


class TestQueens:
    @pytest.mark.parametrize(
        "queen_nb,solution_nb", [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (8, 92), (9, 352)]
    )
    def test_queens_solve(self, queen_nb: int, solution_nb: int) -> None:
        solver = BacktrackSolver(QueensProblem(queen_nb))
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb

    def test_queens_8_solve_ff(self) -> None:
        solver = BacktrackSolver(
            QueensProblem(8), VariableHeuristic(smallest_domain_var_heuristic, min_value_dom_heuristic)
        )
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 92


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    args = parser.parse_args()
    solver = BacktrackSolver(
        QueensProblem(args.n), VariableHeuristic(first_not_instantiated_var_heuristic, min_value_dom_heuristic)
    )
    solver.solve_all()
    stats_print(solver.statistics)
