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
from nucs.examples.car_sequencing.car_sequencing_problem import CarSequencingProblem
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.solvers.backtrack_solver import BacktrackSolver

# Classic CSPLIB instance: 10 cars, 5 options, 6 classes
# Run with:
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.car_sequencing
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    args = parser.parse_args()
    problem = CarSequencingProblem(
        car_nb=10,
        option_nb=5,
        class_nb=6,
        max_per_block=[1, 2, 1, 2, 1],
        block_size=[2, 3, 3, 5, 5],
        demands=[1, 1, 2, 2, 2, 2],
        requires=[
            [1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
    )
    solver = BacktrackSolver(problem, log_level=args.log_level, stks_max_height=args.cp_max_height)
    if args.find_all:
        solver.solve_all()
        if args.display_stats:
            solver.print_statistics()
    else:
        solution = solver.find_one()
        if args.display_stats:
            solver.print_statistics()
        if args.display_solutions:
            problem.print_solution(solution)
