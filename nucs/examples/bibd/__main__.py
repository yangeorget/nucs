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

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS
from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MAX_VALUE

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int)
    parser.add_argument("-b", type=int)
    parser.add_argument("-r", type=int)
    parser.add_argument("-k", type=int)
    parser.add_argument("-l", type=int)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    args = parser.parse_args()
    problem = BIBDProblem(args.v, args.b, args.r, args.k, args.l, args.symmetry_breaking)
    solver = BacktrackSolver(problem, dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE, log_level=args.log_level)
    solver.solve_all()
    print(solver.get_statistics())
