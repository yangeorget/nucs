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

from nucs.examples.default_argument_parser import DefaultArgumentParser, run_solver, solver_kwargs_from_args
from nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem import (
    SportsTournamentSchedulingProblem,
)
from nucs.heuristics.heuristics import VAR_HEURISTIC_DOM_WDEG
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.sports_tournament_scheduling -n 8
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=8)
    args = parser.parse_args()
    problem = SportsTournamentSchedulingProblem(args.n, args.symmetry_breaking)
    run_solver(
        BacktrackSolver(
            problem,
            # last-conflict (measured 2026-11-03): 477 choices instead of 965 at n=8, and 7,193 instead of 10,490
            # at n=10; tests/examples/test_sports_tournament_scheduling.py checks the 477
            **solver_kwargs_from_args(
                args, searches=[Search(var_heuristic=VAR_HEURISTIC_DOM_WDEG)], last_conflict=True
            ),
        ),
        args,
    )
