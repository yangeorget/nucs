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
from rich import print

from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic, max_value_dom_heuristic
from nucs.statistics import get_statistics

# NUMBA_CACHE_DIR=.numba/cache PYTHON_PATH=. python -m nucs.examples.knapsack
if __name__ == "__main__":
    problem = KnapsackProblem(
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        55,
    )
    solver = BacktrackSolver(
        problem, var_heuristic=first_not_instantiated_var_heuristic, dom_heuristic=max_value_dom_heuristic
    )
    solution = solver.maximize(problem.weight)
    print(get_statistics(solver.statistics))
    print(solution)
