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
from nucs.examples.default_argument_parser import DefaultArgumentParser, solver_kwargs_from_args
from nucs.examples.golomb.golomb_problem import GolombProblem, golomb_consistency_algorithm
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import register_consistency_algorithm
from nucs.solvers.multiprocessing_solver import MultiprocessingSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10 --symmetry_breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = GolombProblem(args.n, args.symmetry_breaking)
    golomb_consistency_algorithm = register_consistency_algorithm(golomb_consistency_algorithm)
    kwargs = solver_kwargs_from_args(args, consistency_algorithm=golomb_consistency_algorithm)
    solver = (
        MultiprocessingSolver([BacktrackSolver(prob, **kwargs) for prob in problem.split(args.processors, 0)])
        if args.processors is not None and args.processors > 1
        else BacktrackSolver(problem, **kwargs)
    )
    solution = solver.minimize(problem.length_idx, mode=args.optimization_mode)
    if args.display_stats:
        solver.print_statistics()
    if args.display_solutions:
        problem.print_solution(solution)
