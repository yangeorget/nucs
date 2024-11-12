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
# Copyright 2024 - Yan Georget
###############################################################################
import pytest

from nucs.constants import STATS_IDX_SOLVER_SOLUTION_NB
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN


class TestQueens:
    @pytest.mark.parametrize(
        "queen_nb,solution_nb", [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (7, 40), (8, 92), (9, 352)]
    )
    def test_queens_solve(self, queen_nb: int, solution_nb: int) -> None:
        problem = QueensProblem(queen_nb)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb

    def test_queens_8_solve_ff(self) -> None:
        problem = QueensProblem(8)
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE
        )
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 92
