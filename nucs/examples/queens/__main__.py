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
from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 10
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    kwargs = solver_kwargs_from_args(args)
    solver = (
        MultiprocessingSolver([BacktrackSolver(problem, **kwargs) for problem in problem.split(args.processors, 0)])
        if args.processors is not None and args.processors > 1
        else BacktrackSolver(problem, **kwargs)
    )
    run_solver(solver, args)
