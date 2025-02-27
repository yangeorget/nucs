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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_equiv_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 1


@njit(cache=True)
def get_triggers_equiv_eq(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_equiv_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements b <=> x = c.
    :param domains: the domains of the variables, b is the first domain, x is the second domain
    :param parameters: c is the first parameter
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    b = domains[0]
    x = domains[1]
    c = parameters[0]
    if b[MIN] == 0 and b[MAX] == 0:
        if x[MIN] == c:
            x[MIN] = c + 1
            if x[MIN] > x[MAX]:
                return PROP_INCONSISTENCY
        if x[MAX] == c:
            x[MAX] = c - 1
            if x[MIN] > x[MAX]:
                return PROP_INCONSISTENCY
    elif b[MIN] == 1 and b[MAX] == 1:
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
