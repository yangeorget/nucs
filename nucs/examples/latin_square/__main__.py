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
import argparse

from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.problems.latin_square_problem import LatinSquareProblem, LatinSquareRCProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.latin_square -n 10
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--rc", type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    problem = LatinSquareRCProblem(args.n) if args.rc else LatinSquareProblem(range(args.n))
    BacktrackSolver(problem, args).run(args)
