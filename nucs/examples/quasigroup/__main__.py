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
import argparse

from rich import print

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblemRC
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_SHAVING
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--idempotent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shaving", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--processors", type=int, default=1)
    parser.add_argument("--kind", type=int, choices=[3, 4, 5, 6, 7], default=5)
    args = parser.parse_args()
    problem = QuasigroupProblemRC(args.kind, args.n, args.idempotent, args.symmetry_breaking)
    solver = (
        MultiprocessingSolver([
            BacktrackSolver(
                problem,
                decision_domains=list(range(0, args.n * args.n)),
                consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
                var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
                dom_heuristic_idx=DOM_HEURISTIC_SPLIT_LOW,
                log_level=args.log_level,
            )
            for problem in problem.split(args.processors, 1)
        ])
        if args.processors > 1
        else
        BacktrackSolver(
            problem,
            decision_domains=list(range(0, args.n * args.n)),
            consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
            var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
            dom_heuristic_idx=DOM_HEURISTIC_SPLIT_LOW,
            log_level=args.log_level,
        )
    )
    solver.find_one()
    print(solver.get_statistics())
