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

from rich import print

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS, PB_MASTER, PB_NONE
from nucs.examples.schur_lemma.schur_lemma_problem import SchurLemmaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma -n 20 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--progress_bar", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    problem = SchurLemmaProblem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(problem, log_level=args.log_level, pb_mode=PB_MASTER if args.progress_bar else PB_NONE)
    solver.solve_all()
    print(solver.get_statistics_as_dictionary())
