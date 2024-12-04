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
from nucs.heuristics.heuristics import DOM_HEURISTIC_MID_VALUE
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestMidValueDomHeuristic:
    def test_find_all(self) -> None:
        problem = Problem([(1, 8)])
        solver = BacktrackSolver(problem, dom_heuristic_idx=DOM_HEURISTIC_MID_VALUE)
        solutions = solver.find_all()
        assert len(solutions) == 8
        assert solutions == [[4], [2], [1], [3], [6], [5], [7], [8]]
