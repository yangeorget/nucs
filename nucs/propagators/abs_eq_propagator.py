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
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_abs_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_abs_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    if variable == 0:
        return EVENT_MASK_MIN_MAX
    return EVENT_MASK_MAX


@njit(cache=True)
def compute_domains_abs_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`abs(y)=x`.

    :param domains: the domains of the variables, y is the first domain, x the second
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    y = domains[0]
    x = domains[1]
    # Three cases on the sign of y. When y is strictly positive (resp. negative) abs is monotone,
    # so x and y are tied together: the mirrored assignments below leave x and y with identical
    # bounds, hence testing x alone suffices for inconsistency and entailment (entailment as soon
    # as x is bound, which avoids being re-woken for nothing).
    if y[DOMAIN_MIN] > 0:
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
    elif y[DOMAIN_MAX] < 0:
        if -y[DOMAIN_MAX] > x[DOMAIN_MIN]:
            x[DOMAIN_MIN] = -y[DOMAIN_MAX]
        elif -x[DOMAIN_MIN] < y[DOMAIN_MAX]:
            y[DOMAIN_MAX] = -x[DOMAIN_MIN]
        if -y[DOMAIN_MIN] < x[DOMAIN_MAX]:
            x[DOMAIN_MAX] = -y[DOMAIN_MIN]
        elif -x[DOMAIN_MAX] > y[DOMAIN_MIN]:
            y[DOMAIN_MIN] = -x[DOMAIN_MAX]
        if x[DOMAIN_MIN] > x[DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        if x[DOMAIN_MIN] == x[DOMAIN_MAX]:
            return PROP_ENTAILMENT
    else:
        # 0 lies in y's range: x ranges in [0, max(-y[MIN], y[MAX])] and y in [-x[MAX], x[MAX]].
        # Here y drives the result, so entailment is reported once y is bound.
        x[DOMAIN_MIN] = max(x[DOMAIN_MIN], 0)
        max_y = max(-y[DOMAIN_MIN], y[DOMAIN_MAX])
        x[DOMAIN_MAX] = min(x[DOMAIN_MAX], max_y)
        if x[DOMAIN_MIN] > x[DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        y[DOMAIN_MIN] = max(y[DOMAIN_MIN], -x[DOMAIN_MAX])
        y[DOMAIN_MAX] = min(y[DOMAIN_MAX], x[DOMAIN_MAX])
        if y[DOMAIN_MIN] == y[DOMAIN_MAX]:
            return PROP_ENTAILMENT
    return PROP_CONSISTENCY
