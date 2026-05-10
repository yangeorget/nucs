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
from nucs.examples.social_golfers.social_golfers_problem import SocialGolfersProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.social_golfers
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    args = parser.parse_args()
    parser.add_argument("-n", type=int, default=2)
    parser.add_argument("-s", type=int, default=2)
    parser.add_argument("-w", type=int, default=3)
    problem = SocialGolfersProblem(args.n, args.s, args.w, args.symmetry_breaking)
    run_solver(BacktrackSolver(problem, **solver_kwargs_from_args(args)), args)
