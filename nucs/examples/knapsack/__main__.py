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
import json

from nucs.examples.default_argument_parser import DefaultArgumentParser, solver_kwargs_from_args
from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--dataset", default="datasets/knapsack/simple.json")
    args = parser.parse_args()
    with open(args.dataset, "r") as json_file:
        dataset = json.load(json_file)
        problem = KnapsackProblem(dataset)
        solver = BacktrackSolver(problem, **solver_kwargs_from_args(args, dom_heuristic=DOM_HEURISTIC_MAX_VALUE))
        solution = solver.maximize(problem.weight, mode=args.optimization_mode)
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            problem.print_solution(solution)
