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

from nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem import (
    SportsTournamentSchedulingProblem,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python -m nucs.examples.sports_tournament_scheduling -n 8
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    problem = SportsTournamentSchedulingProblem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    solution = solver.solve_one()
    print(get_statistics(solver.statistics))
    print(problem.solution_as_matrix(solution))
