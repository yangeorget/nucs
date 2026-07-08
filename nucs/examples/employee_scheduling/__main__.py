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
from nucs.examples.default_argument_parser import DefaultArgumentParser, run_optimizer, solver_kwargs_from_args
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
        **solver_kwargs_from_args(
            args, decision_variables=problem.requested_shifts, dom_heuristic=DOM_HEURISTIC_MAX_VALUE
        ),
    )
    run_optimizer(solver, args, problem.satisfied_request_nb, maximize=True)
