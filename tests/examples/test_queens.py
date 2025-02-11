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
from nucs.examples.queens.queens_problem import QueensDualProblem, QueensProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestQueens:
    @pytest.mark.parametrize(
        "dual,var_heuristic,queen_nb,solution_nb",
        [
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 1, 1),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 2, 0),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 3, 0),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 4, 2),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 5, 10),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 6, 4),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 7, 40),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 8, 92),
            (False, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 9, 352),
            (False, VAR_HEURISTIC_SMALLEST_DOMAIN, 8, 92),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 1, 1),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 2, 0),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 3, 0),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 4, 2),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 5, 10),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 6, 4),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 7, 40),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 8, 92),
            (True, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, 9, 352),
            (True, VAR_HEURISTIC_SMALLEST_DOMAIN, 8, 92),
        ],
    )
    def test_queens_solve(self, dual: bool, var_heuristic: int, queen_nb: int, solution_nb: int) -> None:
        problem = QueensDualProblem(queen_nb) if dual else QueensProblem(queen_nb)
        solver = BacktrackSolver(problem, var_heuristic_idx=var_heuristic)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb

    def test_queens_solve_2(self) -> None:
        problem = QueensProblem(2)
        solver = BacktrackSolver(problem)
        solver.find_one()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 0
