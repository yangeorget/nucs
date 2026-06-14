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


def get_complexity_value_precede(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n


@njit(cache=True, fastmath=True)
def get_triggers_value_precede(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX  # any bound change may move the first s-candidate


@njit(cache=True, fastmath=True)
def compute_domains_value_precede(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements value precedence: whenever some x_i equals t, an earlier x_j equals s. Equivalently, the
    first occurrence of s comes before the first occurrence of t (or t does not occur).

    The value t is forbidden in the prefix up to and including the first position whose domain still allows
    s (no s can occur strictly before it). Interior occurrences of t cannot be removed from NuCS interval
    domains, but such a violation is detected as soon as the variable is fixed to t.

    :param domains: the domains of the variables (the array x)
    :type domains: NDArray
    :param parameters: the parameters, s is the first parameter, t is the second parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    s = parameters[0]
    t = parameters[1]
    for i in range(n):
        lo = domains[i, MIN]
        hi = domains[i, MAX]
        # t is forbidden at this position: no value s can precede it yet
        if lo == t and hi == t:
            return PROP_INCONSISTENCY
        if lo == t:
            lo += 1
            domains[i, MIN] = lo
        elif hi == t:
            hi -= 1
            domains[i, MAX] = hi
        # the first position that can still hold s ends the forbidden prefix
        if lo <= s <= hi:
            if lo == s and hi == s:  # s is fixed here, so it precedes every later t
                return PROP_ENTAILMENT
            return PROP_CONSISTENCY
    return PROP_CONSISTENCY
