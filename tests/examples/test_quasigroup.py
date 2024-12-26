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
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestQuasigroup:
    @pytest.mark.parametrize(
        "kind, size, idempotent, solution_nb",
        [
            (5, 6, True, 0),
            (5, 7, True, 3),
            (5, 8, True, 1),
            (5, 9, True, 0),
            (5, 10, True, 0),
            # (5, 11, True, 5),
            # (5, 12, True, 0),
        ],
    )
    def test_qg(self, kind: int, size: int, idempotent: bool, solution_nb: int) -> None:
        problem = QuasigroupProblem(kind, size, idempotent, True)
        solver = BacktrackSolver(
            problem, decision_domains=list(range(0, size * size)), var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN
        )
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb
