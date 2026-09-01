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

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_if_then_else(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables (2 * branches + 1)
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n


@njit(cache=True)
def get_triggers_if_then_else(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever any bound changes.

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
def compute_domains_if_then_else(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the if-then-else selection y = x[k] where k is the smallest index such that the condition
    c[k] holds (the MiniZinc else branch is a literal-true condition, so a branch is normally always taken;
    when every condition is false nothing constrains y, matching the standard decomposition).

    The first b variables are the conditions c (booleans 0/1), the next b are the branch values x, and the
    last is the result y. Filtering is bound-consistent and iterated to a fixpoint (a decision on the first
    still-possible branch can advance which branch is first, re-opening deductions).

    :param domains: the domains of the variables, the b conditions then the b values then y
    :type domains: NDArray
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = (len(domains) - 1) // 2
    y = domains[2 * b]
    # skip the leading branches whose condition is already false: they can never be the first true one
    lo = 0
    while lo < b and domains[lo, DOMAIN_MAX] == 0:
        lo += 1
    if lo == b:
        # no condition can hold -> no branch is taken -> y is unconstrained and stays so
        return PROP_ENTAILMENT
    x_lo = domains[b + lo]
    if domains[lo, DOMAIN_MIN] == 1:
        # the first still-possible condition is true -> branch lo is taken -> y == x[lo]
        new_min = max(x_lo[DOMAIN_MIN], y[DOMAIN_MIN])
        new_max = min(x_lo[DOMAIN_MAX], y[DOMAIN_MAX])
        if new_min > new_max:
            return PROP_INCONSISTENCY
        y[DOMAIN_MIN] = new_min
        y[DOMAIN_MAX] = new_max
        x_lo[DOMAIN_MIN] = new_min
        x_lo[DOMAIN_MAX] = new_max
        # branch lo is fixed as the taken one; the constraint reduces to the equality y == x[lo],
        # entailed once both are ground (then it can no longer be violated)
        return PROP_ENTAILMENT if new_min == new_max else PROP_CONSISTENCY
    # the first still-possible condition c[lo] is unfixed: the constraint entails c[lo] -> (y == x[lo])
    if y[DOMAIN_MIN] == y[DOMAIN_MAX] and x_lo[DOMAIN_MIN] == x_lo[DOMAIN_MAX] and y[DOMAIN_MIN] != x_lo[DOMAIN_MIN]:
        # y and x[lo] are ground and disagree, so branch lo cannot be the taken one -> c[lo] = 0
        domains[lo, DOMAIN_MAX] = 0
        return PROP_CONSISTENCY
    # value deduction: if some later condition is already true, one branch in [lo, hi] is definitely
    # taken; if every candidate value x[lo..hi] is ground to the same v, then y = v whichever is taken
    hi = lo
    while hi < b and domains[hi, DOMAIN_MIN] == 0:
        hi += 1
    if hi < b:
        v = x_lo[DOMAIN_MIN]
        all_same = x_lo[DOMAIN_MIN] == x_lo[DOMAIN_MAX]
        k = lo + 1
        while all_same and k <= hi:
            x_k = domains[b + k]
            if x_k[DOMAIN_MIN] != x_k[DOMAIN_MAX] or x_k[DOMAIN_MIN] != v:
                all_same = False
            k += 1
        if all_same:
            if y[DOMAIN_MIN] > v or y[DOMAIN_MAX] < v:
                return PROP_INCONSISTENCY
            if y[DOMAIN_MIN] != v or y[DOMAIN_MAX] != v:
                y[DOMAIN_MIN] = v
                y[DOMAIN_MAX] = v
                return PROP_CONSISTENCY
    return PROP_CONSISTENCY
