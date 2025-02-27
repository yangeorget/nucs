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
import argparse

from nucs.constants import PB_MASTER, PB_NONE, PB_SLAVE
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MID_VALUE,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_SHAVING
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 10
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--ff", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    solver = (
        MultiprocessingSolver(
            [
                BacktrackSolver(
                    problem,
                    consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
                    var_heuristic_idx=(
                        VAR_HEURISTIC_SMALLEST_DOMAIN if args.ff else VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
                    ),
                    pb_mode=PB_SLAVE if args.progress_bar else PB_NONE,
                    log_level=args.log_level,
                )
                for problem in problem.split(args.processors, 0)
            ],
            pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        )
        if args.processors > 1
        else BacktrackSolver(
            problem,
            consistency_alg_idx=CONSISTENCY_ALG_SHAVING if args.shaving else CONSISTENCY_ALG_BC,
            var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN if args.ff else VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
            dom_heuristic_idx=DOM_HEURISTIC_MID_VALUE,
            pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
            log_level=args.log_level,
            stks_max_height=2048,
        )
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
