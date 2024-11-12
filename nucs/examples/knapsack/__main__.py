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
import argparse

from rich import print

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS
from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    args = parser.parse_args()
    problem = KnapsackProblem(
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        55,
    )
    solver = BacktrackSolver(
        problem,
        var_heuristic_idx=VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE,
        log_level=args.log_level,
    )
    solution = solver.maximize(problem.weight)
    print(solver.get_statistics())
    print(solution)
