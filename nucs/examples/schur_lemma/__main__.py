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
from nucs.examples.schur_lemma.schur_lemma_problem import SchurLemmaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache PYTHON_PATH=. python -m nucs.examples.schur_lemma -n 20
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()
    problem = SchurLemmaProblem(args.n)
    solver = BacktrackSolver(problem)
    solver.solve_all()
    print(get_statistics(solver.statistics))
