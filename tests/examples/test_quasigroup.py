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
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestQuasigroup:
    @pytest.mark.parametrize(
        "kind, size, idempotent, solution_nb",
        [
            (3, 5, True, 0),
            (3, 6, True, 0),
            (3, 7, True, 0),
            (3, 8, True, 18),
            (4, 5, True, 3),
            (4, 6, True, 0),
            (4, 7, True, 0),
            (4, 8, True, 0),
            (5, 5, True, 1),
            (5, 6, True, 0),
            (5, 7, True, 3),
            (5, 8, True, 1),
            (5, 9, True, 0),
            (5, 10, True, 0),
            (5, 11, True, 5),
            # (5, 12, True, 0),      4s
            # (5, 13, True, 0),     97s
            # (5, 14, True, 0),   1380s
            # (5, 15, True, 0),   8h (6p)
            # (5, 16, True, 0),   ?
            # (5, 17, True, 0),   ?
            # (5, 18, True, 0),   ? open
            (6, 5, True, 0),
            (6, 6, True, 0),
            (6, 7, True, 0),
            (6, 8, True, 2),
            (6, 9, True, 4),
            (7, 5, True, 3),
            (7, 6, True, 0),
            (7, 7, True, 0),
            (7, 8, True, 0),
            # (7, 9, True, 1),
        ],
    )
    def test_qg(self, kind: int, size: int, idempotent: bool, solution_nb: int) -> None:
        problem = QuasigroupProblem(kind, size, idempotent, True)
        solver = BacktrackSolver(
            problem,
            decision_domains=list(range(0, size * size)),
            var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
            dom_heuristic_idx=DOM_HEURISTIC_SPLIT_LOW,
        )
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == solution_nb
