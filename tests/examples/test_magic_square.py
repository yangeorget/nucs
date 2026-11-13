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
# Copyright 2024-2026 - Yan Georget
###############################################################################
import pytest

from nucs.examples.magic_square.magic_square_problem import MagicSquareProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search
from nucs.statistics import STATS_IDX_SOLUTION_NB
from tests.marks import SLOW_WITHOUT_JIT


class TestMagicSquare:
    def test_first_diag(self) -> None:
        assert list(MagicSquareProblem(3).first_diag()) == [0, 4, 8]

    def test_second_diag(self) -> None:
        assert list(MagicSquareProblem(3).second_diag()) == [6, 4, 2]

    @pytest.mark.parametrize("size,solution_nb", [(2, 0), (3, 1), pytest.param(4, 880, marks=SLOW_WITHOUT_JIT)])
    def test_solve_all(self, size: int, solution_nb: int) -> None:
        problem = MagicSquareProblem(size)
        solver = BacktrackSolver(
            problem,
            searches=[Search(var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic=DOM_HEURISTIC_MAX_VALUE)],
        )
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
