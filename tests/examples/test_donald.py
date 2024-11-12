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
from nucs.examples.donald.donald_problem import DonaldProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN


class TestDonald:
    def test_donald(self) -> None:
        problem = DonaldProblem()
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE
        )
        solutions = solver.find_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1
        assert solutions[0].tolist() == [4, 3, 5, 9, 1, 8, 6, 2, 7, 0]
