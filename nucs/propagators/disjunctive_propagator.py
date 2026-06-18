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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY

# A sentinel smaller than any earliest start time, used to seed earliest-completion-time accumulations.
MINUS_INF = -(1 << 62)


def get_complexity_disjunctive(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n * n


@njit(cache=True, fastmath=True)
def get_triggers_disjunctive(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever a bound of a start-time variable changes.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index, unused here
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def _filter_est(est: NDArray, lct: NDArray, p: NDArray, n: int) -> bool:
    """
    Raises the earliest start times by overload checking and edge finding on a unary resource.

    For every window upper bound ``L`` (each task's latest completion time) let ``Theta`` be the tasks that
    must complete by ``L`` (``lct_i <= L``). If their earliest completion time ``ECT(Theta)`` exceeds ``L``
    the resource is overloaded. Otherwise, for any task ``t`` that may finish after ``L``, if ``Theta`` plus
    ``t`` cannot all complete by ``L`` then ``t`` must run after every task of ``Theta``, so its earliest
    start is raised to ``ECT(Theta)``.

    ``ECT`` of a set is the greedy earliest-start-ordered chain, which is exact for a unary resource.

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lct: the latest completion times
    :type lct: NDArray
    :param p: the durations
    :type p: NDArray
    :param n: the number of tasks
    :type n: int

    :return: False when the resource is overloaded (inconsistent), True otherwise
    :rtype: bool
    """
    est_order = np.argsort(est)
    new_est = est.copy()
    for c in range(n):
        bound = lct[c]
        # ECT(Theta) with Theta = {i : lct[i] <= bound}, tasks visited in earliest-start order.
        ect = MINUS_INF
        for idx in range(n):
            i = est_order[idx]
            if lct[i] <= bound:
                ect = (est[i] if est[i] > ect else ect) + p[i]
        if ect > bound:
            return False  # overload: Theta cannot complete by bound
        for t in range(n):
            if lct[t] > bound:
                # ECT(Theta ∪ {t}); t is inserted at its earliest-start position.
                ect_t = MINUS_INF
                for idx in range(n):
                    i = est_order[idx]
                    if lct[i] <= bound or i == t:
                        ect_t = (est[i] if est[i] > ect_t else ect_t) + p[i]
                if ect_t > bound and ect > new_est[t]:
                    new_est[t] = ect
    for i in range(n):
        est[i] = new_est[i]
    return True


@njit(cache=True, fastmath=True)
def compute_domains_disjunctive(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the disjunctive (unary resource) constraint: tasks with start times ``domains`` and constant
    durations ``parameters`` must not overlap in time, i.e. for all i != j either ``s_i + p_i <= s_j`` or
    ``s_j + p_j <= s_i``.

    Filtering is overload checking plus edge finding: earliest start times are raised, and latest start
    times are lowered by running the same reasoning on the time-reversed problem.

    :param domains: the domains of the start-time variables, one per task
    :type domains: NDArray
    :param parameters: the durations, one constant per task in the same order as the variables
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    if n <= 1:
        return PROP_ENTAILMENT
    est = np.empty(n, dtype=np.int64)
    lct = np.empty(n, dtype=np.int64)
    p = np.empty(n, dtype=np.int64)
    bound_nb = 0
    for i in range(n):
        est[i] = domains[i, MIN]
        p[i] = parameters[i]
        lct[i] = domains[i, MAX] + p[i]
        if domains[i, MIN] == domains[i, MAX]:
            bound_nb += 1
    # Raise earliest start times.
    if not _filter_est(est, lct, p, n):
        return PROP_INCONSISTENCY
    # Lower latest completion times by mirroring time (est' = -lct, lct' = -est) and reusing the est filter.
    mest = np.empty(n, dtype=np.int64)
    mlct = np.empty(n, dtype=np.int64)
    for i in range(n):
        mest[i] = -lct[i]
        mlct[i] = -est[i]
    if not _filter_est(mest, mlct, p, n):
        return PROP_INCONSISTENCY
    for i in range(n):
        lct[i] = -mest[i]
    for i in range(n):
        if est[i] > domains[i, MIN]:
            domains[i, MIN] = est[i]
        new_max = lct[i] - p[i]
        if new_max < domains[i, MAX]:
            domains[i, MAX] = new_max
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
    # When every start time is already fixed and consistent, the constraint can no longer be violated.
    if bound_nb == n:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
