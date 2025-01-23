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
from nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem import (
    SportsTournamentSchedulingProblem,
)
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.sports_tournament_scheduling -n 8
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress_bar", type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    problem = SportsTournamentSchedulingProblem(args.n, args.symmetry_breaking)
    solver = BacktrackSolver(
        problem,
        var_heuristic_idx=VAR_HEURISTIC_SMALLEST_DOMAIN,
        dom_heuristic_idx=DOM_HEURISTIC_MIN_VALUE,
        pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        log_level=args.log_level,
    )
    solution = solver.find_one()
    print(solver.get_statistics_as_dictionary())
    if solution is not None:
        print(problem.solution_as_matrix(solution))
