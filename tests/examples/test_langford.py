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

from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.langford.langford_problem import LangfordProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestLangford:
    @pytest.mark.parametrize(
        "k, n, solution_nb",
        [
            (2, 3, 2),
            (2, 4, 2),
            (2, 5, 0),
            (2, 6, 0),
            (3, 3, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (3, 9, 6),
        ],
    )
    def test_langford(self, k: int, n: int, solution_nb: int) -> None:
        problem = LangfordProblem(k, n)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
