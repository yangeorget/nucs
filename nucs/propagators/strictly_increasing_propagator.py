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


def get_complexity_strictly_increasing(n: int, parameters: NDArray) -> int:
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


@njit(cache=True)
def get_triggers_strictly_increasing(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_strictly_increasing(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x_i < x_{i+1}` for all i.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    # forward sweep: each x_i is strictly above its predecessor's minimum
    for i in range(1, n):
        domains[i][DOMAIN_MIN] = max(domains[i][DOMAIN_MIN], domains[i - 1][DOMAIN_MIN] + 1)
    # backward sweep: each x_i is strictly below its successor's maximum
    for i in range(n - 2, -1, -1):
        domains[i][DOMAIN_MAX] = min(domains[i][DOMAIN_MAX], domains[i + 1][DOMAIN_MAX] - 1)
    # consistency check and entailment detection (entailed when every pair is already strictly ordered)
    entailed = True
    for i in range(n):
        if domains[i][DOMAIN_MIN] > domains[i][DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        if i < n - 1 and domains[i][DOMAIN_MAX] >= domains[i + 1][DOMAIN_MIN]:
            entailed = False
    return PROP_ENTAILMENT if entailed else PROP_CONSISTENCY
