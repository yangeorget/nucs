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
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_LOW, VAR_HEURISTIC_DOM_WDEG
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup -n 10 --symmetry-breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--idempotent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kind", type=int, choices=[3, 4, 5, 6, 7], default=5)
    args = parser.parse_args()
    problem = QuasigroupProblem(args.kind, args.n, args.idempotent, args.symmetry_breaking)
    # dom/wdeg is the most robust variable heuristic over the kinds (measured 2026-11-03, all pairs of heuristics on
    # kinds 4, 5, 6 and 7, 2 or 3 sizes of each). On kind 5, smallest_domain is faster: 13 ms instead of 20 ms at
    # n=10, 30 ms instead of 77 ms at n=11, 594 ms instead of 2,363 ms at n=12 (--var-heuristic SMALLEST_DOMAIN).
    # On the other kinds it is slower: on kind 6, 649 ms instead of 220 ms at n=11, and no result in 30 s at n=12
    # (dom/wdeg: 20 s); on kind 7, 9.4 s instead of 8.7 s at n=10. On kinds 4, 6 and 7, no other pair of heuristics
    # is faster on all the sizes, and none solves kind 4 at n=10 in 10 s.
    run_solver(
        BacktrackSolver(
            problem,
            **solver_kwargs_from_args(
                args,
                searches=[
                    Search(
                        decision_variables=range(args.n * args.n),
                        var_heuristic=VAR_HEURISTIC_DOM_WDEG,
                        dom_heuristic=DOM_HEURISTIC_SPLIT_LOW,
                    )
                ],
            ),
        ),
        args,
    )
