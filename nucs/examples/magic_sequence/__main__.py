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

from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import last_not_instantiated_var_heuristic, min_value_dom_heuristic
from nucs.statistics import get_statistics

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python -m nucs.examples.magic_sequence -n 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()
    problem = MagicSequenceProblem(args.n)
    solver = BacktrackSolver(
        problem, var_heuristic=last_not_instantiated_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    solver.solve_all()
    print(get_statistics(solver.statistics))
