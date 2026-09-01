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

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_inverse(n: int, parameters: NDArray) -> int:
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
def get_triggers_inverse(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param variable: the index of the variable
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_inverse(domains: NDArray, parameters: NDArray) -> int:
    """
    Channels two inverse arrays next and prev of equal length: prev[j] = i iff next[i] = j.

    Each array's values are the node labels of the *other* array's index set, and the two index sets are
    numbered independently: next's i-th variable stands for node ``next_offset + i`` of prev's numbering and
    prev's j-th for node ``prev_offset + j`` of next's. Both default to 0, so the plain 0-based channelling
    needs no parameter; supplying the offsets lets a differently numbered pair (two 1-based MiniZinc arrays,
    say) be channelled without shifting every variable into an auxiliary one. Both arrays are first trimmed
    to their node labels, which is what bounds the indices this propagator derives from their values.

    :param domains: the domains of the variables, the next variables then the prev variables
    :type domains: NDArray
    :param parameters: the offset of the values next takes then the offset of the values prev takes, or no
        parameter at all when both are 0
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) >> 1
    next = domains[:n]
    prev = domains[n:]
    next_offset = int(parameters[0]) if len(parameters) > 0 else 0
    prev_offset = int(parameters[1]) if len(parameters) > 1 else 0
    if not trim_domains_inverse(n, next, next_offset) or not trim_domains_inverse(n, prev, prev_offset):
        return PROP_INCONSISTENCY
    return (
        PROP_CONSISTENCY
        if filter_domains_inverse(n, next, prev, next_offset, prev_offset)
        and filter_domains_inverse(n, prev, next, prev_offset, next_offset)
        else PROP_INCONSISTENCY
    )


@njit(cache=True)
def trim_domains_inverse(n: int, variables: NDArray, offset: int) -> bool:
    # An inverse array only ever takes the n node labels offset..offset + n - 1.
    for i in range(n):
        variables[i, DOMAIN_MIN] = max(variables[i, DOMAIN_MIN], offset)
        variables[i, DOMAIN_MAX] = min(variables[i, DOMAIN_MAX], offset + n - 1)
        if variables[i, DOMAIN_MIN] > variables[i, DOMAIN_MAX]:
            return False
    return True


@njit(cache=True)
def filter_domains_inverse(n: int, next: NDArray, prev: NDArray, next_offset: int, prev_offset: int) -> bool:
    # next and prev are inverse: prev[j] = i iff next[i] = j, where the node j is the value j + next_offset
    # of next and the node i is the value i + prev_offset of prev. So prev[j] can take the value of node i
    # only when j's label belongs to next[i]'s domain; the test below means prev[j] != node i.
    # Since prev[j]'s feasible values form a contiguous run, we trim its domain from both ends with two
    # pointers, touching only the infeasible prefix and suffix instead of scanning the whole range.
    for j in range(n):
        label = j + next_offset
        lo = prev[j, DOMAIN_MIN] - prev_offset
        hi = prev[j, DOMAIN_MAX] - prev_offset
        if lo == hi:  # prev[j] is fixed, propagate it to next
            next[lo] = label
        else:
            # raise the lower bound past the leading i where prev[j] != i (j is outside next[i])
            while lo <= hi and (label < next[lo, DOMAIN_MIN] or label > next[lo, DOMAIN_MAX]):
                lo += 1
            if lo > hi:  # no feasible value left
                return False
            prev[j, DOMAIN_MIN] = lo + prev_offset
            # lower the upper bound past the trailing i where prev[j] != i
            while label < next[hi, DOMAIN_MIN] or label > next[hi, DOMAIN_MAX]:
                hi -= 1
            prev[j, DOMAIN_MAX] = hi + prev_offset
    return True
