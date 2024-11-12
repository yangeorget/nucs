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
from nucs.constants import STATS_IDX_SOLVER_SOLUTION_NB
from nucs.examples.alpha.alpha_problem import AlphaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN


class TestAlpha:
    def test_alpha(self) -> None:
        problem = AlphaProblem()
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE
        )
        solutions = solver.find_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1
        assert solutions[0][:26].tolist() == [
            5,
            13,
            9,
            16,
            20,
            4,
            24,
            21,
            25,
            17,
            23,
            2,
            8,
            12,
            10,
            19,
            7,
            11,
            15,
            3,
            1,
            26,
            6,
            22,
            14,
            18,
        ]
