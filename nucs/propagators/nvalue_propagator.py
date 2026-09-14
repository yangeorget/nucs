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
from nucs.propagators.alldifferent_propagator import argsort_into, argsort_into_warm

STATE_REPORT = 0  # the engine's change-report cell, which must be the first cell of the hint suffix
STATE_COLD = 1  # 0 until this block has been used once, so a zeroed block seeds the permutations
STATE_ORDER = 2  # the two sort permutations, n of each, by upper bound then by lower bound


def get_state_nvalue(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the change-report cell, a cold flag and the two sort
    permutations this propagator used to rebuild with np.argsort on every call.

    A stale permutation is still a permutation, so the whole block is an untrailed hint and nothing here
    needs restoring on backtrack -- the same bargain alldifferent makes, and the sorts are warm-started from
    it by the same helper. Both orders are over the x_i only, so there are n - 1 of each.

    :param n: the number of variables, one more than the number of x_i
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 2 + 2 * (n - 1))
    :rtype: tuple[int, int]
    """
    return 0, 2 + 2 * max(n - 1, 0)


def get_complexity_nvalue(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n


@njit(cache=True)
def get_triggers_nvalue(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_nvalue(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`y = |\\{x_0, ..., x_{n-1}\\}|`, the number of distinct values taken by the x_i.

    Full domain consistency is NP-hard, so this bounds the count variable y between the maximum number of
    pairwise-disjoint domains (a guaranteed-distinct lower bound) and ``min(n, |union of the domains|)``, and
    handles the all-equal case y = 1 exactly.

    :param domains: the domains of the variables, x is the first n-1 domains, y is the last domain
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) - 1
    x = domains[:n]
    y = domains[n]
    if n == 0:  # no variables: zero distinct values
        if y[DOMAIN_MIN] > 0 or y[DOMAIN_MAX] < 0:
            return PROP_INCONSISTENCY
        y[DOMAIN_MIN] = 0
        y[DOMAIN_MAX] = 0
        return PROP_CONSISTENCY
    # the two orders used to come from a fresh np.argsort each, which is two allocations per call and a
    # sort seeded from scratch; they are kept in the state block instead and re-sorted from their own
    # previous contents, which costs the inversions since the last call rather than a full sort
    order_max = prop_state[STATE_ORDER : STATE_ORDER + n]
    order_min = prop_state[STATE_ORDER + n : STATE_ORDER + 2 * n]
    if prop_state[STATE_COLD] == 0:  # cold: the block is zeroed at solver init, so 0 means never used
        prop_state[STATE_COLD] = 1
        argsort_into(order_max, x, DOMAIN_MAX)
        argsort_into(order_min, x, DOMAIN_MIN)
    else:
        argsort_into_warm(order_max, x, DOMAIN_MAX)
        argsort_into_warm(order_min, x, DOMAIN_MIN)
    # lower bound: the maximum number of pairwise-disjoint domains must take distinct values
    # (interval-scheduling greedy over domains sorted by their upper bound)
    low = 0
    last_end = 0
    for k in range(n):
        i = order_max[k]
        if low == 0 or x[i, DOMAIN_MIN] > last_end:
            low += 1
            last_end = x[i, DOMAIN_MAX]
    # upper bound: at most min(n, number of integers in the union of the domains) distinct values exist
    union = 0
    cur_lo = x[order_min[0], DOMAIN_MIN]
    cur_hi = x[order_min[0], DOMAIN_MAX]
    for k in range(1, n):
        i = order_min[k]
        if x[i, DOMAIN_MIN] > cur_hi + 1:  # gap -> close the current merged block
            union += cur_hi - cur_lo + 1
            cur_lo = x[i, DOMAIN_MIN]
            cur_hi = x[i, DOMAIN_MAX]
        elif x[i, DOMAIN_MAX] > cur_hi:
            cur_hi = x[i, DOMAIN_MAX]
    union += cur_hi - cur_lo + 1
    up = min(union, n)
    changed = False
    if low > y[DOMAIN_MIN]:
        y[DOMAIN_MIN] = low
        changed = True
    if up < y[DOMAIN_MAX]:
        y[DOMAIN_MAX] = up
        changed = True
    if y[DOMAIN_MIN] > y[DOMAIN_MAX]:
        return PROP_INCONSISTENCY
    if low == up:
        # every assignment left in these domains has exactly this many distinct values, and y is now fixed
        # to it: the constraint cannot be violated in this subtree. Both bounds are monotone -- narrowing
        # only makes more domains pairwise disjoint and only shrinks their union -- so this cannot come undone
        return PROP_ENTAILMENT
    if y[DOMAIN_MAX] == 1:  # a single distinct value: every x_i must be equal
        lo = x[0, DOMAIN_MIN]
        hi = x[0, DOMAIN_MAX]
        for i in range(1, n):
            lo = max(lo, x[i, DOMAIN_MIN])
            hi = min(hi, x[i, DOMAIN_MAX])
        if lo > hi:
            return PROP_INCONSISTENCY
        for i in range(n):
            if x[i, DOMAIN_MIN] != lo or x[i, DOMAIN_MAX] != hi:
                x[i, DOMAIN_MIN] = lo
                x[i, DOMAIN_MAX] = hi
                changed = True
    if not changed:
        prop_state[STATE_REPORT] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
