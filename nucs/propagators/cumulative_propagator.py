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
from collections.abc import Sequence

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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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
                    work = min(work, p[j])
                    left = est[j] + p[j] - t1
                    work = min(work, left)
                    right = t2 - lst[j]
                    work = min(work, right)
                    if work > 0:
                        energy += h[j] * work
            if energy > cap_energy:
                return False  # overload
            for i in range(n):
                if h[i] == 0 or p[i] == 0:
                    continue
                work_i = length
                work_i = min(work_i, p[i])
                left = est[i] + p[i] - t1
                work_i = min(work_i, left)
                right = t2 - lst[i]
                work_i = min(work_i, right)
                work_i = max(work_i, 0)
                avail = cap_energy - (energy - h[i] * work_i)  # energy this interval leaves for task i
                if avail < 0:
                    continue
                slack = avail // h[i]
                # raise the earliest start if task i, left-shifted, would not fit
                left_intersection = min(est[i] + p[i], t2) - max(est[i], t1)
                if left_intersection > 0 and h[i] * left_intersection > avail:
                    raised = t2 - slack
                    new_est[i] = max(new_est[i], raised)
                # lower the latest start if task i, right-shifted, would not fit
                right_intersection = min(lst[i] + p[i], t2) - max(lst[i], t1)
                if right_intersection > 0 and h[i] * right_intersection > avail:
                    lowered = t1 + slack - p[i]
                    new_lst[i] = min(new_lst[i], lowered)
    for i in range(n):
        est[i] = new_est[i]
        lst[i] = new_lst[i]
    return True


@njit(cache=True)
def _filter_starts(est: NDArray, lst: NDArray, p: NDArray, h: NDArray, n: int, capacity: int) -> bool:
    """
    Runs timetabling and energetic reasoning on the start bounds until a full sweep changes nothing.

    Timetabling raises starts so that no task overlaps an instant already saturated by the other tasks'
    compulsory parts (and lowers latest starts by mirroring time); energetic reasoning then, over a quadratic
    set of intervals, compares each task's minimum mandatory energy against the capacity. Neither rule is
    idempotent, so a single sweep is not a fixpoint either; the propagator is registered as non-idempotent and
    the engine reschedules it after any sweep that changed a bound.

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lst: the latest start times, lowered in place
    :type lst: NDArray
    :param p: the durations (for a variable-duration task, its minimum, which is sound for the compulsory part)
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
    mest = np.empty(n, dtype=np.int64)
    mlst = np.empty(n, dtype=np.int64)
    # raise earliest start times
    if not _filter_est(est, lst, p, h, n, capacity):
        return False
    # lower latest start times by mirroring time (a start s maps to -(s + p)) and reusing the filter
    for i in range(n):
        mest[i] = -(lst[i] + p[i])
        mlst[i] = -(est[i] + p[i])
    if not _filter_est(mest, mlst, p, h, n, capacity):
        return False
    for i in range(n):
        lst[i] = -mest[i] - p[i]
    # energetic reasoning: stronger interval-based filtering of both bounds
    if not _filter_energetic(est, lst, p, h, n, capacity):
        return False
    for i in range(n):
        if est[i] > lst[i]:  # the start window has emptied
            return False
    return True


def is_vacuous_cumulative(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    """
    Returns whether the parameters make the constraint vacuous, whatever the domains.

    The resource usage at any instant is at most the sum of every demand, so a capacity that already covers
    that sum can never be exceeded, whatever the starts. Tasks of zero duration never occupy any instant, so
    a problem made only of those leaves the usage at 0 everywhere, which a non-negative capacity covers.

    :param n: the number of variables, one start per task
    :type n: int
    :param parameters: the durations, then the demands, then the capacity
    :type parameters: Sequence[int]
    :param domains: the initial domains, unused here
    :type domains: Sequence[tuple[int, int]]

    :return: True when no assignment can violate the constraint
    :rtype: bool
    """
    capacity = parameters[2 * n]
    if sum(parameters[n : 2 * n]) <= capacity:
        return True
    return capacity >= 0 and all(parameters[i] == 0 for i in range(n))


@njit(cache=True)
def compute_domains_cumulative(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the cumulative constraint: tasks with start times ``domains`` run for constant durations and
    consume constant amounts of a resource of fixed capacity; at no instant may the total consumption of the
    tasks in progress exceed the capacity.

    Filtering combines timetabling and energetic reasoning (see :func:`_filter_starts`). Both rules are
    incomplete, so the propagator may stay consistent on an infeasible instance -- that is sound.

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
    if not _filter_starts(est, lst, p, h, n, capacity):
        return PROP_INCONSISTENCY
    ground_nb = 0
    for i in range(n):
        domains[i, MIN] = max(domains[i, MIN], est[i])
        domains[i, MAX] = min(domains[i, MAX], lst[i])
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
        if domains[i, MIN] == domains[i, MAX]:
            ground_nb += 1
    # When every start time is fixed and the profile fits, the constraint can no longer be violated.
    if ground_nb == n:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY


def get_complexity_cumulative_var(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the variable-duration propagator as an int.

    :param n: the number of variables (starts and durations)
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    tasks = n // 2
    return tasks * tasks * tasks


@njit(cache=True)
def get_triggers_cumulative_var(n: int, variable: int, parameters: NDArray) -> int:
    """
    Triggered whenever a start bound or a duration bound changes (a rising duration minimum grows a
    compulsory part).

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


def is_vacuous_cumulative_var(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    """
    Returns whether the parameters make the variable-duration constraint vacuous, whatever the domains.

    As for the constant-duration variant, a capacity covering the sum of every demand can never be exceeded.
    The durations are variables here, so the zero-duration case is not a static property and is not tested.

    :param n: the number of variables, one start and one duration per task
    :type n: int
    :param parameters: the demands, then the capacity
    :type parameters: Sequence[int]
    :param domains: the initial domains, unused here
    :type domains: Sequence[tuple[int, int]]

    :return: True when no assignment can violate the constraint
    :rtype: bool
    """
    tasks = n // 2
    return sum(parameters[:tasks]) <= parameters[tasks]


@njit(cache=True)
def compute_domains_cumulative_var(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the cumulative constraint with variable durations and constant demands and capacity.

    The first ``n`` domains are the start-time variables and the next ``n`` are the duration variables;
    ``parameters = [h_0, ..., h_{n-1}, capacity]``. The compulsory-part and energetic reasoning use each task's
    minimum duration, which is sound (a task is guaranteed to run at least that long), and only the start bounds
    are filtered. When every start and every duration is fixed the minimum duration equals the real one, so a
    ground assignment is checked exactly.

    :param domains: the domains of the start variables then the duration variables
    :type domains: NDArray
    :param parameters: the demands then the capacity
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) // 2
    if n == 0:
        return PROP_ENTAILMENT
    capacity = parameters[n]
    est = np.empty(n, dtype=np.int64)
    lst = np.empty(n, dtype=np.int64)
    p = np.empty(n, dtype=np.int64)
    h = np.empty(n, dtype=np.int64)
    for i in range(n):
        est[i] = domains[i, MIN]
        lst[i] = domains[i, MAX]
        p[i] = domains[n + i, MIN]  # the minimum duration: a sound lower bound on the compulsory part
        h[i] = parameters[i]
        if h[i] > capacity and p[i] > 0:
            return PROP_INCONSISTENCY  # a single task already exceeds the capacity
    if not _filter_starts(est, lst, p, h, n, capacity):
        return PROP_INCONSISTENCY
    for i in range(n):
        domains[i, MIN] = max(domains[i, MIN], est[i])
        domains[i, MAX] = min(domains[i, MAX], lst[i])
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
    ground_nb = 0
    for i in range(2 * n):
        if domains[i, MIN] == domains[i, MAX]:
            ground_nb += 1
    # entail only when every start and every duration is fixed (a free duration can still change the profile)
    if ground_nb == 2 * n:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
