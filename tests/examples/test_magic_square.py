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
from nucs.examples.magic_square.magic_square_problem import MagicSquareProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN


class TestMagicSquare:

    def test_first_diagonal(self) -> None:
        assert MagicSquareProblem(3).first_diag() == [0, 4, 8]

    def test_second_diagonal(self) -> None:
        assert MagicSquareProblem(3).second_diag() == [6, 4, 2]

    @pytest.mark.parametrize("size,solution_nb", [(2, 0), (3, 1), (4, 880)])
    def test_magic_square(self, size: int, solution_nb: int) -> None:
        problem = MagicSquareProblem(size)
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE
        )
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb
