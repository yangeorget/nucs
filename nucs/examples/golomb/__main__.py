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

from nucs.constants import OPTIM_MODES, OPTIM_PRUNE, PB_MASTER, PB_NONE, PB_SLAVE
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--opt_mode", choices=OPTIM_MODES, default=OPTIM_PRUNE)
    args = parser.parse_args()
    problem = GolombProblem(args.n, args.symmetry_breaking)
    consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
    solver = (
        MultiprocessingSolver(
            [
                BacktrackSolver(
                    prob,
                    consistency_alg_idx=consistency_alg_golomb,
                    pb_mode=PB_SLAVE if args.progress_bar else PB_NONE,
                    log_level=args.log_level,
                )
                for prob in problem.split(args.processors, 0)
            ],
            pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        )
        if args.processors > 1
        else BacktrackSolver(
            problem,
            consistency_alg_idx=consistency_alg_golomb,
            pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
            log_level=args.log_level,
        )
    )
    solution = solver.minimize(problem.length_idx, mode=args.opt_mode)
    if args.stats:
        solver.print_statistics()
    if args.display:
        problem.print_solution(solution)
