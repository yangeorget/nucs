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
from nucs.examples.default_argument_parser import DefaultArgumentParser, run_solver, solver_kwargs_from_args
from nucs.examples.langford.langford_problem import LangfordProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.langford
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=12)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    problem = LangfordProblem(args.k, args.n)
    run_solver(
        BacktrackSolver(
            problem,
            **solver_kwargs_from_args(
                args, var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN, dom_heuristic=DOM_HEURISTIC_MAX_VALUE
            ),
        ),
        args,
    )
