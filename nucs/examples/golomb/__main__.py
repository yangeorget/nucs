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
# Copyright 2024 - Yan Georget
###############################################################################
import argparse

from rich import print

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS, OPT_MODES, OPT_PRUNE
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("--opt_mode", choices=OPT_MODES, default=OPT_PRUNE)
    parser.add_argument("--processors", type=int, default=1)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    problem = GolombProblem(args.n, args.symmetry_breaking)
    consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
    solver = (
        MultiprocessingSolver(
            [
                BacktrackSolver(
                    prob, consistency_alg_idx=consistency_alg_golomb, log_level=args.log_level, stack_max_height=128
                )
                for prob in problem.split(args.processors, 0)
            ]
        )
        if args.processors > 1
        else BacktrackSolver(
            problem, consistency_alg_idx=consistency_alg_golomb, log_level=args.log_level, stack_max_height=128
        )
    )
    solution = solver.minimize(problem.length_idx, mode=args.opt_mode)
    print(solver.get_statistics())
    if solution is not None:
        print(solution[: (args.n - 1)])
