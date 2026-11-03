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
from nucs.examples.langford.langford_problem import LangfordProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_MIN_EARLIEST_START
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.searches import Search

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.langford
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=12)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    problem = LangfordProblem(args.k, args.n)
    # the variables are positions: the variable of smallest min at its min fills the sequence from the left, and the
    # propagators see a dense prefix (measured 2026-11-03, against smallest_domain and max_value): 264 ms instead of
    # 735 ms at k=3, n=12, 103 ms instead of 2,075 ms at k=3, n=18, and 20 choices instead of 634 at k=2, n=20;
    # tests/examples/test_langford.py checks the 20
    run_solver(
        BacktrackSolver(
            problem,
            **solver_kwargs_from_args(
                args,
                searches=[
                    Search(var_heuristic=VAR_HEURISTIC_MIN_EARLIEST_START, dom_heuristic=DOM_HEURISTIC_MIN_VALUE)
                ],
            ),
        ),
        args,
    )
