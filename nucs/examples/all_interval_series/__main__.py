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
from nucs.examples.all_interval_series.all_interval_series_problem import AllIntervalSeriesProblem
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.heuristics.heuristics import VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.all_interval_series -n 100
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=8)
    args = parser.parse_args()
    problem = AllIntervalSeriesProblem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(
        problem,
        decision_domains=list(range(args.n)),
        var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
        log_level=args.log_level,
        stks_max_height=2048,
    )
    if args.all:
        solutions = solver.find_all()
        if args.stats:
            solver.print_statistics()
    else:
        solution = solver.find_one()
        if args.stats:
            solver.print_statistics()
        if args.display:
            problem.print_solution(solution)
