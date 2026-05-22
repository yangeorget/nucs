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


def get_complexity_add_c_eq(n: int, parameters: NDArray) -> int:
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


@njit(cache=True, fastmath=True)
def get_triggers_add_c_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_add_c_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x + c = y`.

    :param domains: the domains of the variables, x is the first domain, y the second
    :type domains: NDArray
    :param parameters: the constant c, parameters[0]
    :type parameters: NDArray

    :return: the status of the propagation (consistency or inconsistency) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    c = int(parameters[0])
    y[MIN] = max(y[MIN], x[MIN] + c)
    y[MAX] = min(y[MAX], x[MAX] + c)
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    x[MIN] = y[MIN] - c
    x[MAX] = y[MAX] - c
    if x[MIN] == x[MAX]:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
