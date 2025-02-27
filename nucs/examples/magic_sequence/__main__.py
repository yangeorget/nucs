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
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.magic_sequence.magic_sequence_problem import MagicSequenceProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence -n 100
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()
    problem = MagicSequenceProblem(args.n)
    solver = BacktrackSolver(
        problem,
        decision_domains=list(range(args.n - 1, -1, -1)),
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
