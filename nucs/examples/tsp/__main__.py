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

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS, OPT_MODES, OPT_PRUNE
from nucs.examples.tsp.tsp_instances import TSP_INSTANCES
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.examples.tsp.tsp_var_heuristic import tsp_var_heuristic
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_COST, register_var_heuristic
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_SHAVING
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("--name", choices=["GR17", "GR21", "GR24"], default="GR17")
    parser.add_argument("--opt_mode", choices=OPT_MODES, default=OPT_PRUNE)
    parser.add_argument("--processors", type=int, default=1)
    parser.add_argument("--shaving", type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    tsp_instance = TSP_INSTANCES[args.name]
    n = len(tsp_instance)
    decision_domains = list(range(0, 2 * n))
    problem = TSPProblem(tsp_instance)
    costs = tsp_instance + tsp_instance  # symmetry of costs
    tsp_var_heuristic_idx = register_var_heuristic(tsp_var_heuristic)
    solver = (
        MultiprocessingSolver(
            [
                BacktrackSolver(
                    prob,
                    consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
                    decision_domains=decision_domains,
                    var_heuristic_idx=tsp_var_heuristic_idx,
                    var_heuristic_params=costs,
                    dom_heuristic_idx=DOM_HEURISTIC_MIN_COST,
                    dom_heuristic_params=costs,
                    log_level=args.log_level,
                )
                for prob in problem.split(args.processors, 0)
            ]
        )
        if args.processors > 1
        else BacktrackSolver(
            problem,
            consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
            decision_domains=decision_domains,
            var_heuristic_idx=tsp_var_heuristic_idx,
            var_heuristic_params=costs,
            dom_heuristic_idx=DOM_HEURISTIC_MIN_COST,
            dom_heuristic_params=costs,
            log_level=args.log_level,
        )
    )
    solution = solver.minimize(problem.total_cost, mode=args.opt_mode)
    print(solver.get_statistics())
