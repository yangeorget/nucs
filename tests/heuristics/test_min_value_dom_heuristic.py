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
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestMinValueDomHeuristic:
    def test_find_all(self) -> None:
        problem = Problem([(1, 5)])
        solver = BacktrackSolver(problem, dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE)
        solutions = solver.find_all()
        assert len(solutions) == 5
        assert solutions == [[1], [2], [3], [4], [5]]
