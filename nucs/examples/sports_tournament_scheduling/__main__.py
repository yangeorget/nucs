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
from nucs.heuristics.heuristics import VAR_HEURISTIC_MIN_EARLIEST_START
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
            # the variable of smallest min first, with last-conflict (measured 2026-11-03, against dom/wdeg with
            # last-conflict): 81 choices instead of 477 at n=8, 23 ms instead of 262 ms at n=10, and 0.5 s at n=12
            # and 1.5 s at n=14, where dom/wdeg does not finish in 20 s;
            # tests/examples/test_sports_tournament_scheduling.py checks the 81
            **solver_kwargs_from_args(
                args, searches=[Search(var_heuristic=VAR_HEURISTIC_MIN_EARLIEST_START)], last_conflict=True
            ),
        ),
        args,
    )
