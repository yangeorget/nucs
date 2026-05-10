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
from nucs.examples.all_interval_series.all_interval_series_problem import AllIntervalSeriesProblem
from nucs.examples.default_argument_parser import DefaultArgumentParser, run_solver, solver_kwargs_from_args
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.all_interval_series -n 100
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=8)
    args = parser.parse_args()
    problem = AllIntervalSeriesProblem(args.n, args.symmetry_breaking)
    kwargs = solver_kwargs_from_args(args, decision_variables=range(args.n))
    run_solver(BacktrackSolver(problem, **kwargs), args)
