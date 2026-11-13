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
import argparse

from nucs.examples.default_argument_parser import DefaultArgumentParser, run_solver, solver_kwargs_from_args
from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_SPLIT_HIGH
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence -n 100
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--model-r1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-r2", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    problem = MagicSequenceProblem(args.n, args.model_r1, args.model_r2)
    # most counts are 0: splitting a domain and trying its upper half first removes the large counts that cannot be,
    # where min_value tries the values one by one (measured 2026-11-03): 9 ms instead of 105 ms at n=100, 134 ms
    # instead of 5,850 ms at n=300, and 0.6 s at n=500, where min_value does not finish in 20 s;
    # tests/examples/test_magic_sequence.py checks the 20 choices at n=100
    # after a failure, the search comes back to the variable of the refuted decision: a wrong count of a small value
    # is corrected before the search goes on to the next counts
    searches = [Search(dom_heuristic=DOM_HEURISTIC_SPLIT_HIGH)]
    run_solver(BacktrackSolver(problem, **solver_kwargs_from_args(args, searches=searches, last_conflict=True)), args)
