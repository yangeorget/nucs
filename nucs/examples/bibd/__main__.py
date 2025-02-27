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
# Copyright 2024-2025 - Yan Georget
###############################################################################

from nucs.constants import PB_MASTER, PB_NONE
from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3 --symmetry_breaking
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-v", type=int)
    parser.add_argument("-b", type=int)
    parser.add_argument("-r", type=int)
    parser.add_argument("-k", type=int)
    parser.add_argument("-l", type=int)
    args = parser.parse_args()
    problem = BIBDProblem(args.v, args.b, args.r, args.k, args.l, args.symmetry_breaking)
    solver = BacktrackSolver(
        problem,
        pb_mode=PB_MASTER if args.progress_bar else PB_NONE,
        log_level=args.log_level,
    )
    if args.all:
        solver.solve_all()
        if args.stats:
            solver.print_statistics()
    else:
        solution = solver.find_one()
        if args.stats:
            solver.print_statistics()
        if args.display:
            problem.print_solution(solution)
