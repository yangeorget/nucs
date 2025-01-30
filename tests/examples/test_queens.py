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
# Copyright 2024-2025 - Yan Georget
###############################################################################
import pytest

from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestQueens:
    @pytest.mark.parametrize(
        "var_heuristic,queen_nb,solution_nb",
        [
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 1, 1),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 2, 0),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 3, 0),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 4, 2),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 5, 10),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 6, 4),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 7, 40),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 8, 92),
            (VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 9, 352),
            (VAR_HEURISTIC_SMALLEST_DOMAIN, 8, 92),
        ],
    )
    def test_queens_solve(self, var_heuristic: int, queen_nb: int, solution_nb: int) -> None:
        problem = QueensProblem(queen_nb)
        solver = BacktrackSolver(problem, var_heuristic_idx=var_heuristic)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
