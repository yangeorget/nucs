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
from nucs.examples.schur_lemma.schur_lemma_problem import SchurLemmaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma -n 20 --symmetry_breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()
    problem = SchurLemmaProblem(args.n, args.symmetry_breaking)
    run_solver(BacktrackSolver(problem, **solver_kwargs_from_args(args)), args)
