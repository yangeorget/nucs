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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_member_reif(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the allowed values
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return len(parameters)


@njit(cache=True)
def get_triggers_member_reif(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_member_reif(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`b <=> x \\in \\{a_0, ..., a_{n-1}\\}`.

    Domains are intervals, so filtering is bound-consistent: the allowed values that x can still take form a
    window of the parameters, and only that window's ends can move a bound. Holes between allowed values
    cannot be represented and are therefore left in place, which is also why b is only decided once the
    window is empty (x is never in the set) or covers x's whole interval (x is always in it).

    :param domains: the domains of the variables, b is the first domain, x is the second domain
    :type domains: NDArray
    :param parameters: the allowed values, in strictly ascending order
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    n = len(parameters)
    # The allowed values still in x's range are parameters[lo..hi]; the window is empty when lo > hi.
    lo = 0
    while lo < n and parameters[lo] < x[MIN]:
        lo += 1
    hi = n - 1
    while hi >= 0 and parameters[hi] > x[MAX]:
        hi -= 1
    if lo > hi:  # no allowed value is left in range, so x is never in the set
        if b[MIN] == 1:
            return PROP_INCONSISTENCY
        b[:] = 0
        return PROP_ENTAILMENT
    if hi - lo == x[MAX] - x[MIN]:  # as many allowed values as x has values: x is always in the set
        if b[MAX] == 0:
            return PROP_INCONSISTENCY
        b[:] = 1
        return PROP_ENTAILMENT
    # From x's interval alone both outcomes remain possible, so an unfixed b filters nothing.
    if b[MIN] == 1:  # x is in the set: snap the bounds onto the window's ends
        x[MIN] = parameters[lo]
        x[MAX] = parameters[hi]
        # The window is a run of consecutive integers covering the new interval: never violated again.
        if hi - lo == parameters[hi] - parameters[lo]:
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    if b[MAX] == 0:  # x is not in the set: step each bound past the allowed values it sits on
        while lo <= hi and parameters[lo] == x[MIN]:
            x[MIN] += 1
            lo += 1
        while hi >= lo and parameters[hi] == x[MAX]:
            x[MAX] -= 1
            hi -= 1
        if x[MIN] > x[MAX]:
            return PROP_INCONSISTENCY
        if lo > hi:  # no allowed value is left in range: never violated again
            return PROP_ENTAILMENT
        return PROP_CONSISTENCY
    return PROP_CONSISTENCY
