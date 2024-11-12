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
from nucs.examples.quasigroup.quasigroup_problem import Quasigroup5Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN


class TestQuasigroup:
    @pytest.mark.parametrize(
        "size, solution_nb",
        [
            (7, 3),
            (8, 1),
            (9, 0),
            (10, 0),
            # (11, 5),
            # (12, 0),
        ],
    )
    def test_quasigroup5(self, size: int, solution_nb: int) -> None:
        problem = Quasigroup5Problem(size)
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE
        )
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb
