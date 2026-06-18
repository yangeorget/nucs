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

from nucs.constants import OPTIM_PRUNE
from nucs.examples.default_argument_parser import DefaultArgumentParser, solver_kwargs_from_args
from nucs.examples.jobshop.jobshop_problem import JobShopProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.jobshop --dataset datasets/examples/jobshop/mt06.json
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--dataset", default="datasets/examples/jobshop/mt06.json")
    args = parser.parse_args()
    with open(args.dataset, "r") as json_file:
        dataset = json.load(json_file)
        problem = JobShopProblem(dataset["jobs"])
        kwargs = solver_kwargs_from_args(
            args,
            decision_variables=range(problem.makespan),
            var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
        )
        solver = BacktrackSolver(problem, **kwargs)
        solution = solver.minimize(problem.makespan, mode=args.optimization_mode or OPTIM_PRUNE)
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions and solution is not None:
            print(f"makespan = {solution[problem.makespan]}")
