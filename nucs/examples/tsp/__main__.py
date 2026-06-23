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

from nucs.constants import OPTIM_RESET
from nucs.examples.default_argument_parser import DefaultArgumentParser, solver_kwargs_from_args
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.examples.tsp.tsp_var_heuristic import tsp_var_heuristic
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_COST, register_var_heuristic
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.tsp
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--dataset", default="datasets/examples/tsp/gr17.json")
    args = parser.parse_args()
    with open(args.dataset, "r") as json_file:
        costs = json.load(json_file)["costs"]
        n = len(costs)
        decision_variables = list(range(0, 2 * n))
        problem = TSPProblem(costs)
        costs = costs + costs
        tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
        solver = BacktrackSolver(
            problem,
            **solver_kwargs_from_args(
                args,
                decision_variables=decision_variables,
                var_heuristic=tsp_var_heuristic_idx,
                var_heuristic_params=costs,
                dom_heuristic=DOM_HEURISTIC_MIN_COST,
                dom_heuristic_params=costs,
            ),
        )
        solution = solver.minimize(problem.total_cost, mode=args.optimization_mode or OPTIM_RESET)
        if args.display_stats:
            solver.print_statistics()
