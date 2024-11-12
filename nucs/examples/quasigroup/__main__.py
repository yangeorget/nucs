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
from nucs.examples.quasigroup.quasigroup_problem import Quasigroup5Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    args = parser.parse_args()
    problem = Quasigroup5Problem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(
        problem,
        var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
        dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE,
        log_level=args.log_level,
    )
    solver.solve_all()
    print(solver.get_statistics())
