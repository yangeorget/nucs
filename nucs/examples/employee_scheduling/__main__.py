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
from nucs.constants import OPTIM_MODES, OPTIM_PRUNE
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.employee_scheduling.employee_scheduling_problem import EmployeeSchedulingProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.employee_scheduling
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    parser.add_argument("--opt_mode", choices=OPTIM_MODES, default=OPTIM_PRUNE)
    args = parser.parse_args()
    problem = EmployeeSchedulingProblem()
    solver = BacktrackSolver(problem, log_level=args.log_level)
    solution = solver.maximize(problem.satisfied_request_nb, mode=args.opt_mode)
    if args.stats:
        solver.print_statistics()
    if args.display:
        problem.print_solution(solution)
