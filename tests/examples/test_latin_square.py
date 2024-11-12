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
from nucs.problems.latin_square_problem import LatinSquareProblem, LatinSquareRCProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestLatinSquare:
    @pytest.mark.parametrize(
        "size, solution_nb",
        [
            (1, 1),
            (2, 2),
            (3, 12),
            (4, 576),
            # (5, 161280)
        ],
    )
    def test_latin_square(self, size: int, solution_nb: int) -> None:
        problem = LatinSquareProblem(list(range(size)))
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb

    @pytest.mark.parametrize(
        "size, solution_nb",
        [
            (1, 1),
            (2, 2),
            (3, 12),
            (4, 576),
            # (5, 161280)
        ],
    )
    def test_latin_square_rc(self, size: int, solution_nb: int) -> None:
        problem = LatinSquareRCProblem(size)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb
