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
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestMaxValueDomHeuristic:
    def test_find_all(self) -> None:
        problem = Problem([(1, 5)])
        solver = BacktrackSolver(problem, dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE)
        solutions = solver.find_all()
        assert len(solutions) == 5
        assert solutions == [[5], [4], [3], [2], [1]]
