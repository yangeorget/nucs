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

from nucs.constants import OPTIM_MODES, OPTIM_RESET, PB_MASTER, PB_NONE
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--opt_mode", choices=OPTIM_MODES, default=OPTIM_RESET)
    args = parser.parse_args()
    problem = KnapsackProblem(
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        55,
    )
    solver = BacktrackSolver(
        problem,
        var_heuristic_idx=VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE,
        pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        log_level=args.log_level,
    )
    solution = solver.maximize(problem.weight, mode=args.opt_mode)
    if args.stats:
        solver.print_statistics()
    if args.display:
        problem.print_solution(solution)
