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
from nucs.examples.social_golfers.social_golfers_problem import SocialGolfersProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestGolfers:
    @pytest.mark.parametrize(
        "group_nb, group_size, week_nb, symmetry_breaking, solution_nb",
        [(2, 2, 3, True, 1), (3, 2, 5, True, 2), (3, 3, 4, True, 12)],
    )
    def test_golfers(
        self, group_nb: int, group_size: int, week_nb: bool, symmetry_breaking: bool, solution_nb: int
    ) -> None:
        problem = SocialGolfersProblem(group_nb, group_size, week_nb, symmetry_breaking)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
