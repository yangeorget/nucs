import argparse
from pprint import pprint

import pytest

from nucs.examples.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import STATS_SOLVER_SOLUTION_NB, get_statistics


class TestQueens:
    @pytest.mark.parametrize(
        "queen_nb,solution_nb", [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (7, 40), (8, 92), (9, 352)]
    )
    def test_queens_solve(self, queen_nb: int, solution_nb: int) -> None:
        problem = QueensProblem(queen_nb)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb

    def test_queens_8_solve_ff(self) -> None:
        problem = QueensProblem(8)
        solver = BacktrackSolver(
            problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
        )
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 92


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=12)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    solver = BacktrackSolver(problem)
    solver.solve_all()
    pprint(get_statistics(solver.statistics))
