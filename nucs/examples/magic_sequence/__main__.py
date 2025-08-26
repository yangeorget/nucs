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
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.heuristics.heuristics import VAR_HEURISTIC_FIRST_NOT_INSTANTIATED, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence -n 100
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()
    problem = MagicSequenceProblem(args.n)
    solver = BacktrackSolver(
        problem,
        decision_variables=list(range(args.n - 1, -1, -1)),
        var_heuristic=VAR_HEURISTIC_SMALLEST_DOMAIN if args.ff else VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
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
