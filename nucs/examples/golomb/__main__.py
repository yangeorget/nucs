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

from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache PYTHON_PATH=. python -m nucs.examples.golomb -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    problem = GolombProblem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(problem, consistency_algorithm=golomb_consistency_algorithm)
    solution = solver.minimize(problem.length_idx)
    print(get_statistics(solver.statistics))
    if solution is not None:
        print(solution[: (args.n - 1)])
