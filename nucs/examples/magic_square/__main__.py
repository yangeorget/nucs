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

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS
from nucs.examples.magic_square.magic_square_problem import MagicSquareProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_square -n 4 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    problem = MagicSquareProblem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(
        problem,
        var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
        dom_heuristic_idx=DOM_HEURISTIC_MAX_VALUE,
        log_level=args.log_level,
    )
    solver.solve_all()
    print(solver.get_statistics())
