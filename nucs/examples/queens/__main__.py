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

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic, smallest_domain_var_heuristic
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver
from nucs.statistics import get_statistics

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python -m nucs.examples.queens -n 10
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--processors", type=int, default=1)
    parser.add_argument("--ff", type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    problems = problem.split(args.processors, 0)
    solver = MultiprocessingSolver(
        [
            BacktrackSolver(
                problem,
                var_heuristic=smallest_domain_var_heuristic if args.ff else first_not_instantiated_var_heuristic,
            )
            for problem in problems
        ]
    )
    solver.solve_all()
    print(get_statistics(solver.statistics))
