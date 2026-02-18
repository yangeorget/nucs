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
from nucs.examples.default_argument_parser import DefaultArgumentParser
from nucs.examples.employee_scheduling.employee_scheduling_problem import EmployeeSchedulingProblem
from nucs.heuristics.heuristics import DOM_HEURISTIC_MAX_VALUE
from nucs.solvers.backtrack_solver import BacktrackSolver

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.employee_scheduling
if __name__ == "__main__":
    parser = DefaultArgumentParser()
    args = parser.parse_args()
    problem = EmployeeSchedulingProblem()
    solver = BacktrackSolver(
        problem,
        decision_variables=problem.requested_shifts,
        dom_heuristic=DOM_HEURISTIC_MAX_VALUE,
        log_level=args.log_level,
        stks_max_height=args.cp_max_height,
    )
    solution = solver.maximize(problem.satisfied_request_nb, mode=args.optimization_mode)
    if args.display_stats:
        solver.print_statistics()
    if args.display_solutions:
        problem.print_solution(solution)
