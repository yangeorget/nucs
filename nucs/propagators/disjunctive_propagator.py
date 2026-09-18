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

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)

# The earliest completion time of an empty set of tasks: small enough that adding durations to it stays below any
# real time, large enough that doing so cannot overflow.
NEG = -(1 << 60)

# The sort permutations kept in the state block, per direction (forward, then time-mirrored): the tasks by earliest
# start, by decreasing latest completion, by latest completion, by latest start and by earliest completion.
PERM_EST = 0
PERM_LCT_DESC = 1
PERM_LCT = 2
PERM_LST = 3
PERM_ECT = 4
PERM_NB = 5


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


def get_state_disjunctive(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: a cold flag, then the sort permutations of both directions,
    kept from one call to the next so that each sort starts from the order the previous call left, which bounds
    moving a little between calls barely changes.

    The permutations are untrailed: any permutation is a valid starting point, so a backtrack only makes the next
    sort do more work.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1 + 2 * PERM_NB * n)
    :rtype: tuple[int, int]
    """
    return 0, 1 + 2 * PERM_NB * n


@njit(cache=True)
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


@njit(cache=True)
def _sort(perm: NDArray, keys: NDArray, n: int) -> None:
    """
    Sorts the tasks by key then index, by insertion from the permutation's current order.

    :param perm: the permutation of the tasks, sorted in place
    :type perm: NDArray
    :param keys: the key of each task
    :type keys: NDArray
    :param n: the number of tasks
    :type n: int
    """
    for a in range(1, n):
        task = perm[a]
        key = keys[task]
        b = a - 1
        while b >= 0 and (keys[perm[b]] > key or (keys[perm[b]] == key and perm[b] > task)):
            perm[b + 1] = perm[b]
            b -= 1
        perm[b + 1] = task


@njit(cache=True)
def _rank(est: NDArray, n: int, perm: NDArray, keys: NDArray, rank: NDArray) -> None:
    """
    Sorts the tasks by earliest start and records each task's position, which is its leaf in the trees below.

    :param est: the earliest start times
    :type est: NDArray
    :param n: the number of tasks
    :type n: int
    :param perm: the permutation by earliest start, sorted in place
    :type perm: NDArray
    :param keys: scratch for the sort keys
    :type keys: NDArray
    :param rank: the leaf of each task, written
    :type rank: NDArray
    """
    for i in range(n):
        keys[i] = est[i]
    _sort(perm, keys, n)
    for r in range(n):
        rank[perm[r]] = r


@njit(cache=True)
def _theta_set(size: int, leaf: int, sp: NDArray, ect: NDArray, duration: int, completion: int) -> None:
    """
    Sets a leaf of a Theta-tree and updates its ancestors: a node holds the total duration of its tasks and their
    earliest completion time, ``ect = max(ect_right, ect_left + sp_right)``.

    :param size: the number of leaves, a power of 2
    :type size: int
    :param leaf: the leaf, the task's rank by earliest start
    :type leaf: int
    :param sp: the total durations
    :type sp: NDArray
    :param ect: the earliest completion times
    :type ect: NDArray
    :param duration: the task's duration, 0 to take it out
    :type duration: int
    :param completion: the task's earliest completion, NEG to take it out
    :type completion: int
    """
    k = size + leaf
    sp[k] = duration
    ect[k] = completion
    k >>= 1
    while k >= 1:
        left = k << 1
        right = left + 1
        sp[k] = sp[left] + sp[right]
        ect[k] = max(ect[right], ect[left] + sp[right])
        k >>= 1


@njit(cache=True)
def _theta_clear(size: int, sp: NDArray, ect: NDArray) -> None:
    """
    Empties a Theta-tree.

    :param size: the number of leaves
    :type size: int
    :param sp: the total durations
    :type sp: NDArray
    :param ect: the earliest completion times
    :type ect: NDArray
    """
    for k in range(size << 1):
        sp[k] = 0
        ect[k] = NEG


@njit(cache=True)
def _lambda_pull(k: int, sp: NDArray, ect: NDArray, spb: NDArray, ectb: NDArray, rsp: NDArray, rect: NDArray) -> None:
    """
    Updates a node of a Theta-Lambda tree from its children: the Theta part as in a Theta-tree, and the largest total
    duration and earliest completion reachable by adding at most one gray (Lambda) task, with the gray task
    responsible for each.

    :param k: the node
    :type k: int
    :param sp: the total durations of the Theta tasks
    :type sp: NDArray
    :param ect: the earliest completion times of the Theta tasks
    :type ect: NDArray
    :param spb: the total durations with at most one gray task
    :type spb: NDArray
    :param ectb: the earliest completion times with at most one gray task
    :type ectb: NDArray
    :param rsp: the gray task responsible for spb, or -1
    :type rsp: NDArray
    :param rect: the gray task responsible for ectb, or -1
    :type rect: NDArray
    """
    left = k << 1
    right = left + 1
    sp[k] = sp[left] + sp[right]
    ect[k] = max(ect[right], ect[left] + sp[right])
    a = spb[left] + sp[right]
    b = sp[left] + spb[right]
    if a >= b:
        spb[k] = a
        rsp[k] = rsp[left]
    else:
        spb[k] = b
        rsp[k] = rsp[right]
    x = ectb[right]
    y = ect[left] + spb[right]
    z = ectb[left] + sp[right]
    if x >= y and x >= z:
        ectb[k] = x
        rect[k] = rect[right]
    elif y >= z:
        ectb[k] = y
        rect[k] = rsp[right]
    else:
        ectb[k] = z
        rect[k] = rect[left]


@njit(cache=True)
def _lambda_set(
    size: int,
    leaf: int,
    sp: NDArray,
    ect: NDArray,
    spb: NDArray,
    ectb: NDArray,
    rsp: NDArray,
    rect: NDArray,
    duration: int,
    completion: int,
    gray: int,
) -> None:
    """
    Sets a leaf of a Theta-Lambda tree and updates its ancestors.

    :param size: the number of leaves
    :type size: int
    :param leaf: the leaf
    :type leaf: int
    :param sp: the total durations of the Theta tasks
    :type sp: NDArray
    :param ect: the earliest completion times of the Theta tasks
    :type ect: NDArray
    :param spb: the total durations with at most one gray task
    :type spb: NDArray
    :param ectb: the earliest completion times with at most one gray task
    :type ectb: NDArray
    :param rsp: the gray task responsible for spb, or -1
    :type rsp: NDArray
    :param rect: the gray task responsible for ectb, or -1
    :type rect: NDArray
    :param duration: the gray task's duration, 0 to take the leaf out
    :type duration: int
    :param completion: the gray task's earliest completion, NEG to take the leaf out
    :type completion: int
    :param gray: the gray task, or -1 to take the leaf out
    :type gray: int
    """
    k = size + leaf
    sp[k] = 0
    ect[k] = NEG
    spb[k] = duration
    ectb[k] = completion
    rsp[k] = gray
    rect[k] = gray
    k >>= 1
    while k >= 1:
        _lambda_pull(k, sp, ect, spb, ectb, rsp, rect)
        k >>= 1


@njit(cache=True)
def _edge_finding(
    est: NDArray,
    lct: NDArray,
    p: NDArray,
    n: int,
    size: int,
    perms: NDArray,
    keys: NDArray,
    rank: NDArray,
    new: NDArray,
    sp: NDArray,
    ect: NDArray,
    spb: NDArray,
    ectb: NDArray,
    rsp: NDArray,
    rect: NDArray,
) -> bool:
    """
    Overload checking and edge finding, in O(n log n) with Vilim's Theta-Lambda tree: raises the earliest starts.

    With Theta the tasks that complete by some latest completion time L, the resource is overloaded when Theta
    cannot complete by L, and a task t that may complete later must follow all of Theta, so start at ECT(Theta),
    when Theta and t together cannot complete by L. Taking the windows L by decreasing value, the tasks completing
    after L are gray, and the gray task responsible for the largest completion time is updated and dropped until
    no gray task overflows L.

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lct: the latest completion times
    :type lct: NDArray
    :param p: the durations
    :type p: NDArray
    :param n: the number of tasks
    :type n: int
    :param size: the number of leaves of the trees, a power of 2 at least n
    :type size: int
    :param perms: this direction's sort permutations
    :type perms: NDArray
    :param keys: scratch for the sort keys
    :type keys: NDArray
    :param rank: scratch for the leaf of each task
    :type rank: NDArray
    :param new: scratch for the new bounds
    :type new: NDArray
    :param sp: the tree's total durations
    :type sp: NDArray
    :param ect: the tree's earliest completion times
    :type ect: NDArray
    :param spb: the tree's total durations with a gray task
    :type spb: NDArray
    :param ectb: the tree's earliest completion times with a gray task
    :type ectb: NDArray
    :param rsp: the gray task responsible for spb
    :type rsp: NDArray
    :param rect: the gray task responsible for ectb
    :type rect: NDArray

    :return: False when the resource is overloaded, True otherwise
    :rtype: bool
    """
    _rank(est, n, perms[PERM_EST], keys, rank)
    for k in range(size << 1):
        sp[k] = spb[k] = 0
        ect[k] = ectb[k] = NEG
        rsp[k] = rect[k] = -1
    for i in range(n):
        k = size + rank[i]
        sp[k] = spb[k] = p[i]
        ect[k] = ectb[k] = est[i] + p[i]
    for k in range(size - 1, 0, -1):
        _lambda_pull(k, sp, ect, spb, ectb, rsp, rect)
    for i in range(n):
        keys[i] = -lct[i]
        new[i] = est[i]
    order = perms[PERM_LCT_DESC]
    _sort(order, keys, n)
    for q in range(n):
        j = order[q]
        bound = lct[j]
        if ect[1] > bound:
            return False
        while ectb[1] > bound:
            i = rect[1]
            new[i] = max(new[i], ect[1])
            _lambda_set(size, rank[i], sp, ect, spb, ectb, rsp, rect, 0, NEG, -1)
        _lambda_set(size, rank[j], sp, ect, spb, ectb, rsp, rect, p[j], est[j] + p[j], j)
    for i in range(n):
        est[i] = new[i]
    return True


@njit(cache=True)
def _not_last(
    est: NDArray,
    lct: NDArray,
    p: NDArray,
    n: int,
    size: int,
    perms: NDArray,
    keys: NDArray,
    rank: NDArray,
    new: NDArray,
    in_theta: NDArray,
    sp: NDArray,
    ect: NDArray,
) -> None:
    """
    Not-last, in O(n log n) with a Theta-tree, as Vilim formulates it and Gecode implements it: lowers the latest
    completion times.

    Taking the tasks i by latest completion, Theta collects the tasks whose latest start is below lct_i. When the
    tasks of Theta other than i cannot all complete before i's latest start, i is not last among them, so it must
    complete by the largest latest start in Theta. The tasks inserted on the way get the same test against Theta.

    :param est: the earliest start times
    :type est: NDArray
    :param lct: the latest completion times, lowered in place
    :type lct: NDArray
    :param p: the durations
    :type p: NDArray
    :param n: the number of tasks
    :type n: int
    :param size: the number of leaves of the tree
    :type size: int
    :param perms: this direction's sort permutations
    :type perms: NDArray
    :param keys: scratch for the sort keys
    :type keys: NDArray
    :param rank: scratch for the leaf of each task
    :type rank: NDArray
    :param new: scratch for the new bounds
    :type new: NDArray
    :param in_theta: scratch for Theta's membership
    :type in_theta: NDArray
    :param sp: the tree's total durations
    :type sp: NDArray
    :param ect: the tree's earliest completion times
    :type ect: NDArray
    """
    _rank(est, n, perms[PERM_EST], keys, rank)
    _theta_clear(size, sp, ect)
    by_lct = perms[PERM_LCT]
    for i in range(n):
        keys[i] = lct[i]
    _sort(by_lct, keys, n)
    by_lst = perms[PERM_LST]
    for i in range(n):
        keys[i] = lct[i] - p[i]
        new[i] = lct[i]
        in_theta[i] = 0
    _sort(by_lst, keys, n)
    q = 0
    for a in range(n):
        i = by_lct[a]
        j = -1
        while q < n and lct[i] > lct[by_lst[q]] - p[by_lst[q]]:
            k = by_lst[q]
            if j >= 0 and ect[1] > lct[k] - p[k] and lct[j] - p[j] < new[k]:
                new[k] = lct[j] - p[j]
            j = k
            _theta_set(size, rank[k], sp, ect, p[k], est[k] + p[k])
            in_theta[k] = 1
            q += 1
        if j >= 0:
            if in_theta[i]:
                _theta_set(size, rank[i], sp, ect, 0, NEG)
                completion = ect[1]
                _theta_set(size, rank[i], sp, ect, p[i], est[i] + p[i])
            else:
                completion = ect[1]
            if completion > lct[i] - p[i] and lct[j] - p[j] < new[i]:
                new[i] = lct[j] - p[j]
    for i in range(n):
        lct[i] = new[i]


@njit(cache=True)
def _detectable_precedences(
    est: NDArray,
    lct: NDArray,
    p: NDArray,
    n: int,
    size: int,
    perms: NDArray,
    keys: NDArray,
    rank: NDArray,
    new: NDArray,
    in_theta: NDArray,
    sp: NDArray,
    ect: NDArray,
) -> None:
    """
    Detectable precedences, in O(n log n) with a Theta-tree: raises the earliest start times.

    A precedence i before j is detectable when j cannot run before i, i.e. when i's latest start is below j's
    earliest completion. Taking the tasks j by earliest completion, Theta collects the tasks whose latest start is
    below it, and j starts no earlier than the earliest completion of Theta without j.

    :param est: the earliest start times, raised in place
    :type est: NDArray
    :param lct: the latest completion times
    :type lct: NDArray
    :param p: the durations
    :type p: NDArray
    :param n: the number of tasks
    :type n: int
    :param size: the number of leaves of the tree
    :type size: int
    :param perms: this direction's sort permutations
    :type perms: NDArray
    :param keys: scratch for the sort keys
    :type keys: NDArray
    :param rank: scratch for the leaf of each task
    :type rank: NDArray
    :param new: scratch for the new bounds
    :type new: NDArray
    :param in_theta: scratch for Theta's membership
    :type in_theta: NDArray
    :param sp: the tree's total durations
    :type sp: NDArray
    :param ect: the tree's earliest completion times
    :type ect: NDArray
    """
    _rank(est, n, perms[PERM_EST], keys, rank)
    _theta_clear(size, sp, ect)
    by_ect = perms[PERM_ECT]
    for i in range(n):
        keys[i] = est[i] + p[i]
    _sort(by_ect, keys, n)
    by_lst = perms[PERM_LST]
    for i in range(n):
        keys[i] = lct[i] - p[i]
        new[i] = est[i]
        in_theta[i] = 0
    _sort(by_lst, keys, n)
    q = 0
    for a in range(n):
        j = by_ect[a]
        completion_j = est[j] + p[j]
        while q < n and lct[by_lst[q]] - p[by_lst[q]] < completion_j:
            i = by_lst[q]
            _theta_set(size, rank[i], sp, ect, p[i], est[i] + p[i])
            in_theta[i] = 1
            q += 1
        if in_theta[j]:
            _theta_set(size, rank[j], sp, ect, 0, NEG)
            completion = ect[1]
            _theta_set(size, rank[j], sp, ect, p[j], est[j] + p[j])
        else:
            completion = ect[1]
        new[j] = max(new[j], completion)
    for j in range(n):
        est[j] = new[j]


@njit(cache=True)
def compute_domains_disjunctive(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements the disjunctive (unary resource) constraint: tasks with start times ``domains`` and constant
    durations ``parameters`` must not overlap in time, i.e. for all i != j either ``s_i + p_i <= s_j`` or
    ``s_j + p_j <= s_i``.

    Filtering combines overload checking, edge finding, not-first/not-last and detectable precedences, each in
    O(n log n) with Vilim's Theta-trees as Gecode does, and each run on the tasks then on their time-mirrored image
    (``est' = -lct``, ``lct' = -est``) to prune the other bound. One pass does not reach the rules' common fixpoint,
    so the propagator is not idempotent and the engine calls it again after a pass that changed a bound. Not-last is
    Gecode's: on jobshop instances under the same search, the trees are Gecode's, failure for failure.

    :param domains: the domains of the start-time variables, one per task
    :type domains: NDArray
    :param parameters: the durations, one constant per task in the same order as the variables
    :type parameters: NDArray
    :param prop_state: this propagator's state block: a cold flag then the sort permutations of both directions
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    if n <= 1:
        return PROP_ENTAILMENT
    perms = prop_state[1:].reshape(2, PERM_NB, n)
    if prop_state[0] == 0:  # cold: the block is zeroed at solver init, so the permutations are not yet any
        prop_state[0] = 1
        for direction in range(2):
            for perm in range(PERM_NB):
                for i in range(n):
                    perms[direction, perm, i] = i
    size = 1
    while size < n:
        size <<= 1
    tree_nb = size << 1
    buffer = np.empty(9 * n + 6 * tree_nb, dtype=np.int64)
    est = buffer[:n]
    lct = buffer[n : 2 * n]
    mest = buffer[2 * n : 3 * n]
    mlct = buffer[3 * n : 4 * n]
    p = buffer[4 * n : 5 * n]
    keys = buffer[5 * n : 6 * n]
    rank = buffer[6 * n : 7 * n]
    new = buffer[7 * n : 8 * n]
    in_theta = buffer[8 * n : 9 * n]
    trees = buffer[9 * n :].reshape(6, tree_nb)
    sp, ect, spb, ectb, rsp, rect = trees[0], trees[1], trees[2], trees[3], trees[4], trees[5]
    bound_nb = 0
    for i in range(n):
        est[i] = domains[i, DOMAIN_MIN]
        p[i] = parameters[i]
        lct[i] = domains[i, DOMAIN_MAX] + p[i]
        if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
            bound_nb += 1
    forward = perms[0]
    mirrored = perms[1]
    if not _edge_finding(est, lct, p, n, size, forward, keys, rank, new, sp, ect, spb, ectb, rsp, rect):
        return PROP_INCONSISTENCY
    for i in range(n):
        mest[i] = -lct[i]
        mlct[i] = -est[i]
    if not _edge_finding(mest, mlct, p, n, size, mirrored, keys, rank, new, sp, ect, spb, ectb, rsp, rect):
        return PROP_INCONSISTENCY
    for i in range(n):
        lct[i] = -mest[i]
    _not_last(est, lct, p, n, size, forward, keys, rank, new, in_theta, sp, ect)
    for i in range(n):
        mest[i] = -lct[i]
        mlct[i] = -est[i]
    _not_last(mest, mlct, p, n, size, mirrored, keys, rank, new, in_theta, sp, ect)  # not-first
    for i in range(n):
        est[i] = -mlct[i]
    _detectable_precedences(est, lct, p, n, size, forward, keys, rank, new, in_theta, sp, ect)
    for i in range(n):
        mest[i] = -lct[i]
        mlct[i] = -est[i]
    _detectable_precedences(mest, mlct, p, n, size, mirrored, keys, rank, new, in_theta, sp, ect)
    for i in range(n):
        lct[i] = -mest[i]
    for i in range(n):
        if est[i] + p[i] > lct[i]:
            return PROP_INCONSISTENCY
    for i in range(n):
        domains[i, DOMAIN_MIN] = max(domains[i, DOMAIN_MIN], est[i])
        domains[i, DOMAIN_MAX] = min(domains[i, DOMAIN_MAX], lct[i] - p[i])
    if bound_nb == n:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
