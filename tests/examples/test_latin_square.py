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

from nucs.problems.latin_square_problem import LatinSquareProblem, LatinSquareRCProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_IDX_SOLUTION_NB


class TestLatinSquare:
    @pytest.mark.parametrize(
        "size, solution_nb, rc",
        [
            (1, 1, False),
            (2, 2, False),
            (3, 12, False),
            (4, 576, False),
            # (5, 161280, False),
            (1, 1, True),
            (2, 2, True),
            (3, 12, True),
            (4, 576, True),
            # (5, 161280, True)
        ],
    )
    def test_solve_all(self, size: int, solution_nb: int, rc: bool) -> None:
        problem = LatinSquareRCProblem(size) if rc else LatinSquareProblem(range(size))
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
