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
from nucs.examples.knapsack.knapsack_datasets import DATASETS
from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestKnapsack:
    def test_knapsack(self) -> None:
        problem = KnapsackProblem(DATASETS["SIMPLE"])
        solver = BacktrackSolver(
            problem, var_heuristic=VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, dom_heuristic=DOM_HEURISTIC_MAX_VALUE
        )
        solution = solver.maximize(problem.weight)
        assert solution is not None
        assert solution[problem.weight] == 54
