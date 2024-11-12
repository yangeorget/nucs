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
from nucs.examples.schur_lemma.schur_lemma_problem import SchurLemmaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestSchurLemma:
    @pytest.mark.parametrize(
        "ball_nb, solution_nb",
        [
            (3, 8),
            (4, 17),
            (5, 31),
            (6, 61),
            (7, 124),
            (8, 145),
            (9, 268),
            (10, 147),
            (11, 91),
            (12, 59),
            (13, 8),
            (14, 0),
            (15, 0),
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
            (20, 0),
        ],
    )
    def test_solve(self, ball_nb: int, solution_nb: bool) -> None:
        problem = SchurLemmaProblem(ball_nb)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb
