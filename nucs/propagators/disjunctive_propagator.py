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
def _filter_not_last(est: NDArray, lct: NDArray, p: NDArray, n: int) -> None:
    """
    Lowers the latest completion times by the not-last rule on a unary resource.

    Task ``t`` is "not last" among ``Theta ∪ {t}`` -- where ``Theta = {i != t : lct_i <= lct_t}`` -- when the
    earliest completion time ``ECT(Theta)`` exceeds ``t``'s latest start ``lct_t - p_t``: ``Theta`` cannot all
    finish before ``t`` would start, so at least one task of ``Theta`` runs after ``t``. Then ``t`` must
    complete before that task starts, hence by the largest latest start of ``Theta``, lowering ``lct_t``.

    Not-first is obtained by mirroring time and calling this routine again (as edge finding does), so a single
    routine covers both rules. Filtering is complementary to edge finding (neither subsumes the other).

    :param est: the earliest start times
    :type est: NDArray
    :param lct: the latest completion times, lowered in place
    :type lct: NDArray
    :param p: the durations
    :type p: NDArray
    :param n: the number of tasks
    :type n: int
    """
    est_order = np.argsort(est)
    new_lct = lct.copy()
    for t in range(n):
        # ECT(Theta) (greedy earliest-start chain) and the largest latest start over Theta.
        ect = MINUS_INF
        max_lst = MINUS_INF
        for idx in range(n):
            i = est_order[idx]
            if i != t and lct[i] <= lct[t]:
                ect = (est[i] if est[i] > ect else ect) + p[i]
                lst_i = lct[i] - p[i]
                if lst_i > max_lst:
                    max_lst = lst_i
        if ect > lct[t] - p[t] and max_lst < new_lct[t]:  # t cannot be last: it must finish by max_lst
            new_lct[t] = max_lst
    for t in range(n):
        lct[t] = new_lct[t]


@njit(cache=True, fastmath=True)
def _filter_detectable_precedences(est: NDArray, lct: NDArray, p: NDArray, n: int) -> None:
    """
    Raises the earliest start times by detectable precedences on a unary resource.

    A precedence ``i ≪ j`` (``i`` before ``j``) is *detectable* when ``j`` cannot run before ``i``, i.e. when
    ``i``'s latest start ``lct_i - p_i`` is below ``j``'s earliest completion ``est_j + p_j``. Task ``j`` then
    runs after all such predecessors, so its earliest start is raised to ``ECT(Pred(j))``.

    Detectable successors (lowering latest completion times) are obtained by mirroring time and calling this
    routine again, as edge finding does. Filtering is complementary to edge finding and not-first/not-last
    (none subsumes another).

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lct: the latest completion times
    :type lct: NDArray
    :param p: the durations
    :type p: NDArray
    :param n: the number of tasks
    :type n: int
    """
    est_order = np.argsort(est)
    new_est = est.copy()
    for j in range(n):
        ect_j = est[j] + p[j]
        # ECT(Pred(j)) with Pred(j) = {i != j : lct_i - p_i < ect_j}, tasks visited in earliest-start order.
        ect = MINUS_INF
        for idx in range(n):
            i = est_order[idx]
            if i != j and lct[i] - p[i] < ect_j:
                ect = (est[i] if est[i] > ect else ect) + p[i]
        if ect > new_est[j]:  # j must start after all its detectable predecessors complete
            new_est[j] = ect
    for j in range(n):
        est[j] = new_est[j]


@njit(cache=True, fastmath=True)
def compute_domains_disjunctive(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the disjunctive (unary resource) constraint: tasks with start times ``domains`` and constant
    durations ``parameters`` must not overlap in time, i.e. for all i != j either ``s_i + p_i <= s_j`` or
    ``s_j + p_j <= s_i``.

    Filtering combines overload checking, edge finding, not-first/not-last and detectable precedences: each
    rule raises earliest start times, and lowers latest start times by running the same reasoning on the
    time-reversed problem. The rules are not individually idempotent, so the whole block is iterated until a
    full sweep changes no bound, leaving the propagator at its own fixpoint.

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
    mest = np.empty(n, dtype=np.int64)
    mlct = np.empty(n, dtype=np.int64)
    prev_est = np.empty(n, dtype=np.int64)
    prev_lct = np.empty(n, dtype=np.int64)
    # The individual rules are not idempotent (a batch pass leaves cascaded pruning on the table), so we
    # iterate the whole filtering block until a full sweep changes no bound: the propagator then returns at
    # its own fixpoint rather than relying on the solver to re-trigger it. Each changing sweep tightens at
    # least one bound by >= 1, so the loop terminates in at most the total initial domain width.
    has_changed = True
    while has_changed:
        for i in range(n):
            prev_est[i] = est[i]
            prev_lct[i] = lct[i]
        # Edge finding: raise earliest start times.
        if not _filter_est(est, lct, p, n):
            return PROP_INCONSISTENCY
        # Lower latest completion times by mirroring time (est' = -lct, lct' = -est) and reusing the filter.
        for i in range(n):
            mest[i] = -lct[i]
            mlct[i] = -est[i]
        if not _filter_est(mest, mlct, p, n):
            return PROP_INCONSISTENCY
        for i in range(n):
            lct[i] = -mest[i]
        # Not-last: lower latest completion times (complementary to edge finding).
        _filter_not_last(est, lct, p, n)
        # Not-first: raise earliest start times by mirroring time and reusing the not-last filter.
        for i in range(n):
            mest[i] = -lct[i]
            mlct[i] = -est[i]
        _filter_not_last(mest, mlct, p, n)
        for i in range(n):
            est[i] = -mlct[i]
        # Detectable precedences: raise earliest start times, then lower latest completion times by mirroring.
        _filter_detectable_precedences(est, lct, p, n)
        for i in range(n):
            mest[i] = -lct[i]
            mlct[i] = -est[i]
        _filter_detectable_precedences(mest, mlct, p, n)
        for i in range(n):
            lct[i] = -mest[i]
        # A task whose earliest start has crossed its latest start cannot be scheduled: inconsistency. This
        # also caps the loop when the rules diverge (e.g. mutually detectable precedences push est upward).
        has_changed = False
        for i in range(n):
            if est[i] + p[i] > lct[i]:
                return PROP_INCONSISTENCY
            if est[i] != prev_est[i] or lct[i] != prev_lct[i]:
                has_changed = True
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
