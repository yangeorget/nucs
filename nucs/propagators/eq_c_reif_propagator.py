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


def get_complexity_eq_c_reif(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return 1


@njit(cache=True)
def get_triggers_eq_c_reif(n: int, variable: int, parameters: NDArray) -> int:
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
def advise_eq_c_reif(domains: NDArray, parameters: NDArray) -> bool:
    """
    Advisor for :math:`b \\Leftrightarrow x = c`: when b is false (x != c) c can be dropped only from a bound;
    when b is true (x = c) x can tighten unless already fixed to c; when b is free b can be decided only when c
    is outside x or x is ground.

    :param domains: the domains of the variables, b is the first domain, x the second
    :type domains: NDArray
    :param parameters: c is the first parameter
    :type parameters: NDArray

    :return: whether the propagator should be scheduled
    :rtype: bool
    """
    b = domains[0]
    x = domains[1]
    c = int(parameters[0])
    if b[MAX] == 0:  # b false: x != c
        return x[MIN] == c or x[MAX] == c
    if b[MIN] == 1:  # b true: x = c
        return x[MIN] != c or x[MAX] != c
    return x[MIN] > c or x[MAX] < c or x[MIN] == x[MAX]  # b free: c outside x or x ground


@njit(cache=True)
def compute_domains_eq_c_reif(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`b <=> x = c`.

    :param domains: the domains of the variables, b is the first domain, x is the second domain
    :type domains: NDArray
    :param parameters: c is the first parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    c = int(parameters[0])
    if b[MAX] == 0:
        if x[MIN] == c:
            x[MIN] = c + 1
            if x[MIN] > x[MAX]:
                return PROP_INCONSISTENCY
        if x[MAX] == c:
            x[MAX] = c - 1
            if x[MIN] > x[MAX]:
                return PROP_INCONSISTENCY
    elif b[MIN] == 1:
        if c < x[MIN] or c > x[MAX]:
            return PROP_INCONSISTENCY
        else:
            x[:] = c
            return PROP_ENTAILMENT
    if x[MIN] > c or x[MAX] < c:
        b[:] = 0
        return PROP_ENTAILMENT
    elif x[MIN] == c and x[MAX] == c:
        b[:] = 1
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
