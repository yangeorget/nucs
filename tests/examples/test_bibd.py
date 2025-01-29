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
from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestBIBD:
    @pytest.mark.parametrize(
        "v, b, r, k, l, solution_nb", [(6, 10, 5, 3, 2, 1), (7, 7, 3, 3, 1, 1), (8, 14, 7, 4, 3, 92)]
    )
    def test_bibd(self, v: int, b: int, r: int, k: int, l: int, solution_nb: int) -> None:
        problem = BIBDProblem(v, b, r, k, l)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
