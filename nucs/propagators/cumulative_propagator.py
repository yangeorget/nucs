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


def get_complexity_cumulative(n: int, parameters: NDArray) -> int:
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
def get_triggers_cumulative(n: int, variable: int, parameters: NDArray) -> int:
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
def _filter_est(est: NDArray, lst: NDArray, p: NDArray, h: NDArray, n: int, capacity: int) -> bool:
    """
    Raises the earliest start times by timetabling on a cumulative resource.

    The compulsory part of a task is the interval ``[lst, est + p)`` it must occupy whatever its start; over
    that interval it consumes ``h`` units. The sum of the compulsory parts is the resource profile. If the
    profile ever exceeds the capacity the resource is overloaded. Otherwise a task may not overlap any instant
    where the profile of the *other* tasks would leave less than its own height free, so its earliest start is
    pushed past every such forbidden region.

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lst: the latest start times
    :type lst: NDArray
    :param p: the durations
    :type p: NDArray
    :param h: the resource demands (heights)
    :type h: NDArray
    :param n: the number of tasks
    :type n: int
    :param capacity: the resource capacity
    :type capacity: int

    :return: False when the resource is overloaded (inconsistent), True otherwise
    :rtype: bool
    """
    # collect the boundaries of the non-empty compulsory parts
    bounds = np.empty(2 * n, dtype=np.int64)
    bound_nb = 0
    for i in range(n):
        ect = est[i] + p[i]
        if p[i] > 0 and h[i] > 0 and lst[i] < ect:  # task i has a compulsory part [lst, ect)
            bounds[bound_nb] = lst[i]
            bounds[bound_nb + 1] = ect
            bound_nb += 2
    if bound_nb == 0:
        return True  # no compulsory part: nothing forces the profile, nothing to filter
    sorted_bounds = np.sort(bounds[:bound_nb])
    # the distinct boundaries delimit the profile segments [seg_start, seg_end)
    seg_start = np.empty(bound_nb, dtype=np.int64)
    seg_end = np.empty(bound_nb, dtype=np.int64)
    seg_nb = 0
    for idx in range(1, bound_nb):
        if sorted_bounds[idx] != sorted_bounds[idx - 1]:
            seg_start[seg_nb] = sorted_bounds[idx - 1]
            seg_end[seg_nb] = sorted_bounds[idx]
            seg_nb += 1
    # the profile height of each segment (sum of the demands of the tasks whose compulsory part covers it)
    seg_height = np.zeros(seg_nb, dtype=np.int64)
    for s in range(seg_nb):
        a = seg_start[s]
        b = seg_end[s]
        total = 0
        for j in range(n):
            if p[j] > 0 and h[j] > 0 and lst[j] <= a and est[j] + p[j] >= b:  # cp of j covers [a, b)
                total += h[j]
        if total > capacity:
            return False  # overload
        seg_height[s] = total
    # push each task past the regions where it would not fit alongside the others' profile
    for i in range(n):
        if h[i] == 0 or p[i] == 0:
            continue
        ect = est[i] + p[i]
        i_has_cp = lst[i] < ect
        tau = est[i]
        for s in range(seg_nb):
            if seg_end[s] <= tau:
                continue
            if seg_start[s] >= tau + p[i]:  # the segment starts after the placement window, none left
                break
            height_without_i = seg_height[s]
            if i_has_cp and lst[i] <= seg_start[s] and ect >= seg_end[s]:  # this segment includes i itself
                height_without_i -= h[i]
            if height_without_i > capacity - h[i]:  # task i cannot overlap this segment
                tau = seg_end[s]
        est[i] = tau
    return True


@njit(cache=True, fastmath=True)
def _filter_energetic(est: NDArray, lst: NDArray, p: NDArray, h: NDArray, n: int, capacity: int) -> bool:
    """
    Filters both start bounds by energetic reasoning on a cumulative resource.

    Over an interval ``[t1, t2)`` a task must spend, whatever its start, a minimum energy
    ``h * max(0, min(t2 - t1, p, ect - t1, t2 - lst))``. If the tasks' minimum energies exceed
    ``capacity * (t2 - t1)`` the interval is overloaded. Otherwise, letting ``avail`` be the energy an interval
    leaves for a task once the others' minima are placed, a task that would consume more than ``avail`` when
    left-shifted (started at ``est``) cannot start that early -- its earliest start is raised to
    ``t2 - avail // h`` -- and symmetrically its latest start is lowered to ``t1 + avail // h - p``.

    Each deduction is sound for any interval, so a subset of intervals only weakens filtering, never soundness.
    The intervals scanned are the pairs ``(t1, t2)``, ``t1 < t2``, with ``t1`` an earliest start, latest start
    or earliest completion and ``t2`` an earliest completion, latest completion or latest start -- the standard
    ``O(n^2)`` set (not the full energetic-reasoning interval set, so filtering is strong but not complete).

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lst: the latest start times, lowered in place
    :type lst: NDArray
    :param p: the durations
    :type p: NDArray
    :param h: the resource demands (heights)
    :type h: NDArray
    :param n: the number of tasks
    :type n: int
    :param capacity: the resource capacity
    :type capacity: int

    :return: False when an interval is overloaded (inconsistent), True otherwise
    :rtype: bool
    """
    lefts = np.empty(3 * n, dtype=np.int64)
    rights = np.empty(3 * n, dtype=np.int64)
    for i in range(n):
        lefts[3 * i] = est[i]
        lefts[3 * i + 1] = lst[i]
        lefts[3 * i + 2] = est[i] + p[i]
        rights[3 * i] = est[i] + p[i]
        rights[3 * i + 1] = lst[i] + p[i]
        rights[3 * i + 2] = lst[i]
    lefts = np.sort(lefts)
    rights = np.sort(rights)
    new_est = est.copy()
    new_lst = lst.copy()
    for li in range(3 * n):
        if li > 0 and lefts[li] == lefts[li - 1]:
            continue
        t1 = lefts[li]
        for ri in range(3 * n):
            if ri > 0 and rights[ri] == rights[ri - 1]:
                continue
            t2 = rights[ri]
            if t2 <= t1:
                continue
            length = t2 - t1
            cap_energy = capacity * length
            # total minimum (mandatory) energy of all tasks over [t1, t2)
            energy = 0
            for j in range(n):
                if p[j] > 0 and h[j] > 0:
                    work = length
                    if p[j] < work:
                        work = p[j]
                    left = est[j] + p[j] - t1
                    if left < work:
                        work = left
                    right = t2 - lst[j]
                    if right < work:
                        work = right
                    if work > 0:
                        energy += h[j] * work
            if energy > cap_energy:
                return False  # overload
            for i in range(n):
                if h[i] == 0 or p[i] == 0:
                    continue
                work_i = length
                if p[i] < work_i:
                    work_i = p[i]
                left = est[i] + p[i] - t1
                if left < work_i:
                    work_i = left
                right = t2 - lst[i]
                if right < work_i:
                    work_i = right
                if work_i < 0:
                    work_i = 0
                avail = cap_energy - (energy - h[i] * work_i)  # energy this interval leaves for task i
                if avail < 0:
                    continue
                slack = avail // h[i]
                # raise the earliest start if task i, left-shifted, would not fit
                left_intersection = min(est[i] + p[i], t2) - max(est[i], t1)
                if left_intersection > 0 and h[i] * left_intersection > avail:
                    raised = t2 - slack
                    if raised > new_est[i]:
                        new_est[i] = raised
                # lower the latest start if task i, right-shifted, would not fit
                right_intersection = min(lst[i] + p[i], t2) - max(lst[i], t1)
                if right_intersection > 0 and h[i] * right_intersection > avail:
                    lowered = t1 + slack - p[i]
                    if lowered < new_lst[i]:
                        new_lst[i] = lowered
    for i in range(n):
        est[i] = new_est[i]
        lst[i] = new_lst[i]
    return True


@njit(cache=True, fastmath=True)
def compute_domains_cumulative(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the cumulative constraint: tasks with start times ``domains`` run for constant durations and
    consume constant amounts of a resource of fixed capacity; at no instant may the total consumption of the
    tasks in progress exceed the capacity.

    Filtering combines timetabling and energetic reasoning: timetabling raises starts so that no task overlaps
    an instant already saturated by the other tasks' compulsory parts (and lowers latest starts by mirroring
    time); energetic reasoning then, over a quadratic set of intervals, compares each task's minimum mandatory
    energy against the capacity to detect overloads and push the bounds further. Neither rule is idempotent, so
    the whole block is iterated until a full sweep changes no bound, leaving the propagator at its own fixpoint.
    Both rules are incomplete, so the propagator may stay consistent on an infeasible instance -- that is sound.

    The parameters pack, in order, the ``n`` durations, then the ``n`` demands (heights), then the capacity:
    ``parameters = [p_0, ..., p_{n-1}, h_0, ..., h_{n-1}, capacity]``.

    :param domains: the domains of the start-time variables, one per task
    :type domains: NDArray
    :param parameters: the durations, the demands and the capacity, as described above
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    if n == 0:
        return PROP_ENTAILMENT
    capacity = parameters[2 * n]
    est = np.empty(n, dtype=np.int64)
    lst = np.empty(n, dtype=np.int64)
    p = np.empty(n, dtype=np.int64)
    h = np.empty(n, dtype=np.int64)
    for i in range(n):
        est[i] = domains[i, MIN]
        lst[i] = domains[i, MAX]
        p[i] = parameters[i]
        h[i] = parameters[n + i]
        if h[i] > capacity and p[i] > 0:
            return PROP_INCONSISTENCY  # a single task already exceeds the capacity
    mest = np.empty(n, dtype=np.int64)
    mlst = np.empty(n, dtype=np.int64)
    prev_est = np.empty(n, dtype=np.int64)
    prev_lst = np.empty(n, dtype=np.int64)
    # Timetabling is not idempotent (raising a start grows a compulsory part, which can push more tasks), so
    # iterate the whole block until a full sweep changes no bound. Each changing sweep tightens at least one
    # bound by >= 1, so the loop terminates in at most the total initial domain width.
    has_changed = True
    while has_changed:
        for i in range(n):
            prev_est[i] = est[i]
            prev_lst[i] = lst[i]
        # raise earliest start times
        if not _filter_est(est, lst, p, h, n, capacity):
            return PROP_INCONSISTENCY
        # lower latest start times by mirroring time (a start s maps to -(s + p)) and reusing the filter
        for i in range(n):
            mest[i] = -(lst[i] + p[i])
            mlst[i] = -(est[i] + p[i])
        if not _filter_est(mest, mlst, p, h, n, capacity):
            return PROP_INCONSISTENCY
        for i in range(n):
            lst[i] = -mest[i] - p[i]
        # energetic reasoning: stronger interval-based filtering of both bounds
        if not _filter_energetic(est, lst, p, h, n, capacity):
            return PROP_INCONSISTENCY
        has_changed = False
        for i in range(n):
            if est[i] > lst[i]:  # the start window has emptied
                return PROP_INCONSISTENCY
            if est[i] != prev_est[i] or lst[i] != prev_lst[i]:
                has_changed = True
    ground_nb = 0
    for i in range(n):
        if est[i] > domains[i, MIN]:
            domains[i, MIN] = est[i]
        if lst[i] < domains[i, MAX]:
            domains[i, MAX] = lst[i]
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
        if domains[i, MIN] == domains[i, MAX]:
            ground_nb += 1
    # When every start time is fixed and the profile fits, the constraint can no longer be violated.
    if ground_nb == n:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
