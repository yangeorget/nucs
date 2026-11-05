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
"""
What the search loops of the solvers share: the statuses that say why a loop returned, the cells of the search
control, and the extraction of a solution. Each solver has its own loop, run_search_loop, in the module of the
solver: backtrack_solver.py for BacktrackSolver, lcg_solver.py for LcgSolver.
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MIN

# Why a search loop returned.
# The trail and the choice point stack are caller-allocated, so they cannot grow inside @njit. Rather than
# sizing them for a worst case that never happens -- depth x (2 x domain_nb + 1) entries, which would
# give back the memory this representation wins -- the search stops and says which one is full, and the
# solver grows it and resumes. Nothing of the search is lost: the state, the trail marks and the
# positions all stay valid across the reallocation.
SOLVER_RUNNING = 0  # nothing filled up: the search returned a solution or exhausted itself
SOLVER_TRAIL_FULL = 1  # the search stopped because the trail needs more room, not because it is over
SOLVER_CHOICE_POINTS_FULL = 2  # likewise for the stack of choice points
SOLVER_INTERRUPTED = 3  # the search stopped because interrupt() asked it to, from outside the compiled loop
SOLVER_RESTART = 4  # the descent reached its failure limit: the solver restarts from the root and resumes

# The cells of search_control, the state of the restarts and of last-conflict that the compiled loop reads and writes.
# It is solver state, not choice-point state: nothing about it is trailed.
SEARCH_CONTROL_RESTART_LIMIT = 0  # the failures after which the descent restarts, -1 for no restart
SEARCH_CONTROL_FAILURE_NB = 1  # the failures since the search last started from the root
SEARCH_CONTROL_LAST_CONFLICT = 2  # 1 when last-conflict is on, 0 otherwise
SEARCH_CONTROL_CONFLICT_VARIABLE = 3  # the variable of the last decision that led to a failure, -1 when none
SEARCH_CONTROL_DECISION_VARIABLE = 4  # the variable of the last decision, -1 once a refutation follows it
SEARCH_CONTROL_WIDTH = 5


@njit(cache=True)
def get_solution(domains: NDArray) -> NDArray:
    """
    Returns the solution to the problem.

    :param domains: the domains, every one of them ground
    :type domains: NDArray

    :return: a Numpy array
    :rtype: NDArray
    """
    return domains[:, DOMAIN_MIN].copy()
