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
from nucs.heuristics.heuristics import DOM_HEURISTIC_MID_VALUE
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestMidValueDomHeuristic:
    def test_find_all(self) -> None:
        problem = Problem([(1, 8)])
        solver = BacktrackSolver(problem, dom_heuristic=DOM_HEURISTIC_MID_VALUE)
        solutions = solver.find_all()
        assert len(solutions) == 8
        assert solutions == [[4], [2], [1], [3], [6], [5], [7], [8]]

    def test_find_all_over_a_negative_domain(self) -> None:
        """A split value below zero has to survive the trip to the solver.

        A domain heuristic is compiled for SIGN_DOM_HEURISTIC and reached through a function pointer, so
        how it hands its value back is an ABI question: returned in a heterogeneous tuple, the int32 half
        is zero-extended and -5 arrives as 4294967291. An int32 pair widens nothing and so avoids that,
        and this pins it -- without a negative domain, the whole suite passes either way.
        """
        problem = Problem([(-5, -1)])
        solver = BacktrackSolver(problem, dom_heuristic=DOM_HEURISTIC_MID_VALUE)
        assert solver.find_all() == [[-3], [-5], [-4], [-2], [-1]]
