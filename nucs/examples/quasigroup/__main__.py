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
import argparse

from nucs.examples.default_argument_parser import DefaultArgumentParser, run_solver, solver_kwargs_from_args
from nucs.examples.quasigroup.quasigroup_problem import QuasigroupProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_LOW
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--idempotent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kind", type=int, choices=[3, 4, 5, 6, 7], default=5)
    args = parser.parse_args()
    problem = QuasigroupProblem(args.kind, args.n, args.idempotent, args.symmetry_breaking)
    run_solver(
        BacktrackSolver(
            problem,
            **solver_kwargs_from_args(
                args,
                decision_variables=range(0, args.n * args.n),
                dom_heuristic=DOM_HEURISTIC_SPLIT_LOW,
            ),
        ),
        args,
    )
