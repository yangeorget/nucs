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

from nucs.constants import EVENT_MASK_MAX, EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_abs_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 1


@njit(cache=True)
def get_triggers_abs_eq(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    if dom_idx == 0:
        return EVENT_MASK_MIN_MAX
    return EVENT_MASK_MAX


@njit(cache=True)
def compute_domains_abs_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements abs(y)=x.
    :param domains: the domains of the variables, y is the first domain, x the second
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    y = domains[0]
    x = domains[1]
    if y[MIN] > 0:
        x[MIN] = max(x[MIN], y[MIN])
        x[MAX] = min(x[MAX], y[MAX])
        if x[MIN] > x[MAX]:
            return PROP_INCONSISTENCY
        y[MIN] = max(y[MIN], x[MIN])
        y[MAX] = min(y[MAX], x[MAX])
        if y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
    elif y[MAX] < 0:
        x[MIN] = max(x[MIN], -y[MAX])
        x[MAX] = min(x[MAX], -y[MIN])
        if x[MIN] > x[MAX]:
            return PROP_INCONSISTENCY
        y[MIN] = max(y[MIN], -x[MAX])
        y[MAX] = min(y[MAX], -x[MIN])
        if y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
    else:
        x[MIN] = max(x[MIN], 0)
        x[MAX] = min(x[MAX], max(-y[MIN], y[MAX]))
        if x[MIN] > x[MAX]:
            return PROP_INCONSISTENCY
        y[MIN] = max(y[MIN], -x[MAX])
        y[MAX] = min(y[MAX], x[MAX])
        if y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
