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
from ncs.statistics import statistics_print, STATS_SOLVER_SOLUTION_NB
from ncs.memory import init_domain_changes


class TestQueens:

    def test_queens_4(self) -> None:
        problem = QueensProblem(4)
        shr_domain_changes = init_domain_changes(4, True)
        assert problem.filter(shr_domain_changes)


    @pytest.mark.parametrize(
        "queen_nb,solution_nb", [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (8, 92), (9, 352)]
    )
    def test_queens_solve(self, queen_nb: int, solution_nb: int) -> None:
        problem = QueensProblem(queen_nb)
        solver = BacktrackSolver(problem)
        solver.find_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb

    def test_queens_8_solve_ff(self) -> None:
        problem = QueensProblem(8)
        solver = BacktrackSolver(problem, VariableHeuristic(smallest_domain_var_heuristic, min_value_dom_heuristic))
        solver.find_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 92


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    solver = BacktrackSolver(problem, VariableHeuristic(first_not_instantiated_var_heuristic, min_value_dom_heuristic))
    solver.find_all()
    statistics_print(problem.statistics)
