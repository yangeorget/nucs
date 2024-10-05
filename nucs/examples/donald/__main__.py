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
# Copyright 2024 - Yan Georget
###############################################################################
from rich import print

from nucs.examples.donald.donald_problem import DonaldProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics

# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache PYTHON_PATH=. python -m nucs.examples.donald
if __name__ == "__main__":
    problem = DonaldProblem()
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    print(get_statistics(solver.statistics))
    for solution in solver.solve():
        print(problem.solution_as_dict(solution))
