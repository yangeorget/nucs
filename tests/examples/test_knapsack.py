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
from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED


class TestKnapsack:
    def test_knapsack(self) -> None:
        problem = KnapsackProblem(
            [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
            [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
            55,
        )
        solver = BacktrackSolver(
            problem, var_heuristic_idx=VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE
        )
        solution = solver.maximize(problem.weight)
        assert solution is not None
        assert solution[problem.weight] == 54
