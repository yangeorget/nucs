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
# Copyright 2024-2025 - Yan Georget
###############################################################################

from nucs.constants import OPTIM_MODES, OPTIM_PRUNE, PB_MASTER, PB_NONE, PB_SLAVE
from nucs.examples.default_argument_parser import DefaultArgumentParser
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
    parser = DefaultArgumentParser()
    parser.add_argument("--name", choices=["GR17", "GR21", "GR24"], default="GR17")
    parser.add_argument("--opt_mode", choices=OPTIM_MODES, default=OPTIM_PRUNE)
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
                    pb_mode=PB_SLAVE if args.progress_bar else PB_NONE,
                )
                for prob in problem.split(args.processors, 0)
            ],
            pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
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
            pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        )
    )
    solution = solver.minimize(problem.total_cost, mode=args.opt_mode)
    if args.stats:
        solver.print_statistics()
