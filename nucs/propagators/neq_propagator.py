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
    EVENT_MASK_GROUND,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_neq(n: int, parameters: NDArray) -> int:
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
def get_triggers_neq(n: int, variable: int, parameters: NDArray) -> int:
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
    return EVENT_MASK_GROUND


@njit(cache=True)
def compute_domains_neq(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`x \\neq y`.

    :param domains: the domains of the variables, x is the first domain, y the second
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    # Disjoint domains: x != y can never be violated.
    if x[DOMAIN_MAX] < y[DOMAIN_MIN] or y[DOMAIN_MAX] < x[DOMAIN_MIN]:
        return PROP_ENTAILMENT
    # Bound consistency can only filter when a variable is bound: the forbidden value is then
    # removed from the other domain, but solely when it sits on one of its bounds (no holes).
    if x[DOMAIN_MIN] == x[DOMAIN_MAX]:
        if y[DOMAIN_MIN] == x[DOMAIN_MIN]:
            if y[DOMAIN_MIN] == y[DOMAIN_MAX]:  # x = y: stepping a bound past an int32 limit overflows
                return PROP_INCONSISTENCY
            y[DOMAIN_MIN] += 1
        elif y[DOMAIN_MAX] == x[DOMAIN_MIN]:
            y[DOMAIN_MAX] -= 1
        else:
            return PROP_CONSISTENCY
        return PROP_ENTAILMENT
    if y[DOMAIN_MIN] == y[DOMAIN_MAX]:  # when a bound of x is y, x is not fixed: the step cannot empty x
        if x[DOMAIN_MIN] == y[DOMAIN_MIN]:
            x[DOMAIN_MIN] += 1
        elif x[DOMAIN_MAX] == y[DOMAIN_MIN]:
            x[DOMAIN_MAX] -= 1
        else:
            return PROP_CONSISTENCY
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
