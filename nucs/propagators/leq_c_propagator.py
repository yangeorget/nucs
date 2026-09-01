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
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_leq_c(n: int, parameters: NDArray) -> int:
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
def get_triggers_leq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN if variable == 0 else EVENT_MASK_MAX


@njit(cache=True)
def compute_domains_leq_c(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x <= y + c`.

    :param domains: the domains of the variables, x is the first domain, y is the second domain
    :type domains: NDArray
    :param parameters: the parameters of the propagator, c is the first parameter
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    c = int(parameters[0])
    if x[DOMAIN_MAX] <= y[DOMAIN_MIN] + c:
        return PROP_ENTAILMENT
    x[DOMAIN_MAX] = min(x[DOMAIN_MAX], y[DOMAIN_MAX] + c)
    if x[DOMAIN_MIN] > x[DOMAIN_MAX]:
        return PROP_INCONSISTENCY
    y[DOMAIN_MIN] = max(y[DOMAIN_MIN], x[DOMAIN_MIN] - c)
    if y[DOMAIN_MIN] > y[DOMAIN_MAX]:
        return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
