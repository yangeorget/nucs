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
from nucs.examples.default_argument_parser import DefaultArgumentParser
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
    solver = BacktrackSolver(
        problem,
        log_level=args.log_level,
        stks_max_height=args.cp_max_height,
    )
    if args.find_all:
        solver.solve_all()
        if args.display_stats:
            solver.print_statistics()
    else:
        solution = solver.find_one()
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            problem.print_solution(solution)
