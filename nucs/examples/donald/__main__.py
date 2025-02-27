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

from nucs.constants import PB_MASTER, PB_NONE
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.donald.donald_problem import DonaldProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.donald
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    args = parser.parse_args()
    problem = DonaldProblem()
    solver = BacktrackSolver(
        problem,
        var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
        dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE,
        pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        log_level=args.log_level,
    )
    if args.all:
        solver.solve_all()
        solver.print_statistics()
    else:
        solution = solver.find_one()
        if args.stats:
            solver.print_statistics()
        if args.display:
            problem.print_solution(solution)
