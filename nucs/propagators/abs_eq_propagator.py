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
from collections.abc import Sequence

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


def get_state_abs_eq(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


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
def compute_domains_abs_eq(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`abs(y)=x`.

    :param domains: the domains of the variables, y is the first domain, x the second
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    y = domains[0]
    x = domains[1]
    # Four bounds are the whole of what this propagator can write, and all of them are already being
    # loaded, so the change is read off a snapshot rather than tracked at each of the eleven write sites
    # scattered over the three sign cases -- fewer places to get wrong, and nothing extra to load.
    y_min, y_max, x_min, x_max = y[DOMAIN_MIN], y[DOMAIN_MAX], x[DOMAIN_MIN], x[DOMAIN_MAX]
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
        # 0 lies in y's range: x ranges in [0, max(-y[DOMAIN_MIN], y[DOMAIN_MAX])] and y in [-x[DOMAIN_MAX], x[DOMAIN_MAX]].
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
    if y[DOMAIN_MIN] == y_min and y[DOMAIN_MAX] == y_max and x[DOMAIN_MIN] == x_min and x[DOMAIN_MAX] == x_max:
        prop_state[0] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
