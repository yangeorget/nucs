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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_permutation_aux(n: int, parameters: NDArray) -> int:
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


@njit(cache=True, fastmath=True)
def get_triggers_permutation_aux(n: int, variable: int, parameters: NDArray) -> int:
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


@njit(cache=True, fastmath=True)
def compute_domains_permutation_aux(domains: NDArray, parameters: NDArray) -> int:
    """
    An auxiliary propagator needed to connect the next and prev variables of a permutation problem.

    :param domains: the domains of the variables, the next and prev variables
    :type domains: NDArray
    :param parameters: the parameters of the propagator, unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) >> 1
    next = domains[:n]
    prev = domains[n:]
    return (
        PROP_CONSISTENCY
        if filter_domains_permutation(n, next, prev) and filter_domains_permutation(n, prev, next)
        else PROP_INCONSISTENCY
    )


@njit(cache=True, fastmath=True)
def filter_domains_permutation(n: int, next: NDArray, prev: NDArray) -> bool:
    # next and prev are inverse: prev[j] = i iff next[i] = j. So prev[j] can take value i only when
    # j belongs to next[i]'s domain; the test (j < next[i, MIN] or j > next[i, MAX]) means prev[j] != i.
    # Since prev[j]'s feasible values form a contiguous run, we trim its domain from both ends with two
    # pointers, touching only the infeasible prefix and suffix instead of scanning the whole range.
    for j in range(n):
        lo = prev[j, MIN]
        hi = prev[j, MAX]
        if lo == hi:  # prev[j] is fixed, propagate it to next
            next[lo] = j
        else:
            # raise the lower bound past the leading i where prev[j] != i (j is outside next[i])
            while lo <= hi and (j < next[lo, MIN] or j > next[lo, MAX]):
                lo += 1
            if lo > hi:  # no feasible value left
                return False
            prev[j, MIN] = lo
            # lower the upper bound past the trailing i where prev[j] != i
            while j < next[hi, MIN] or j > next[hi, MAX]:
                hi -= 1
            prev[j, MAX] = hi
    return True
