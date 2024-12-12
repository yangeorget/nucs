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
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_COST, VAR_HEURISTIC_MAX_REGRET
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_SHAVING

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("--name", choices=["GR17", "GR21", "GR24"], default="GR17")
    parser.add_argument("--shaving", type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opt_mode", choices=OPT_MODES, default=OPT_PRUNE)
    args = parser.parse_args()
    tsp_instance = TSP_INSTANCES[args.name]
    problem = TSPProblem(tsp_instance)
    solver = BacktrackSolver(
        problem,
        consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
        decision_domains=list(range(len(tsp_instance))),
        var_heuristic_idx=VAR_HEURISTIC_MAX_REGRET,  # register_var_heuristic(mrp_var_heuristic),
        var_heuristic_params=tsp_instance,
        dom_heuristic_idx=DOM_HEURISTIC_MIN_COST,  # register_dom_heuristic(mcp_dom_heuristic),
        dom_heuristic_params=tsp_instance,
        log_level=args.log_level,
    )
    solution = solver.minimize(problem.shr_domain_nb - 1, mode=args.opt_mode)
    print(solver.get_statistics())
