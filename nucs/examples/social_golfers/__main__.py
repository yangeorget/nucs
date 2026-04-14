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
from nucs.examples.alpha.alpha_problem import AlphaProblem
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    args = parser.parse_args()
    problem = AlphaProblem()
    solver = BacktrackSolver(
        problem,
        var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN,
        dom_heuristic=DOM_HEURISTIC_MIN_VALUE,
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
