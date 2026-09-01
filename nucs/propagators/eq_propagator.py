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


def get_complexity_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_eq(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x = y`.

    :param domains: the domains of the variables, x is the first domain, y the second
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    # Intersect the two domains: after this x and y share identical bounds, so testing x alone
    # suffices for inconsistency and entailment.
    if y[DOMAIN_MIN] > x[DOMAIN_MIN]:
        x[DOMAIN_MIN] = y[DOMAIN_MIN]
    elif x[DOMAIN_MIN] > y[DOMAIN_MIN]:
        y[DOMAIN_MIN] = x[DOMAIN_MIN]
    if y[DOMAIN_MAX] < x[DOMAIN_MAX]:
        x[DOMAIN_MAX] = y[DOMAIN_MAX]
    elif x[DOMAIN_MAX] < y[DOMAIN_MAX]:
        y[DOMAIN_MAX] = x[DOMAIN_MAX]
    if x[DOMAIN_MIN] > x[DOMAIN_MAX]:
        return PROP_INCONSISTENCY
    if x[DOMAIN_MIN] == x[DOMAIN_MAX]:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
