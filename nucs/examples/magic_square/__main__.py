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
from nucs.examples.magic_square.magic_square_problem import MagicSquareProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_RANDOM_VALUE, VAR_HEURISTIC_DOM_WDEG
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.restarts import RESTART_LUBY, Restarts
from nucs.solvers.search import Search

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_square -n 6 --symmetry-breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=6)
    args = parser.parse_args()
    problem = MagicSquareProblem(args.n, args.symmetry_breaking)
    # dom/wdeg alone can stay in a subtree without a solution: the restarts take it back to the root with the weights
    # that it learned, and the random values make each descent explore a different part of the tree
    searches = [Search(var_heuristic=VAR_HEURISTIC_DOM_WDEG, dom_heuristic=DOM_HEURISTIC_RANDOM_VALUE)]
    kwargs = solver_kwargs_from_args(args, searches=searches, restarts=Restarts(RESTART_LUBY, 100))
    run_solver(BacktrackSolver(problem, **kwargs), args)
