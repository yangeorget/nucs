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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_max_leq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


@njit(cache=True)
def get_triggers_max_leq(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN if dom_idx < n - 1 else EVENT_MASK_MAX


@njit(cache=True)
def compute_domains_max_leq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Max_i x_i <= x_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    x = domains[:-1]
    y = domains[-1]
    if np.max(x[:, MAX]) <= y[MIN]:
        return PROP_ENTAILMENT
    y[MIN] = max(y[MIN], np.max(x[:, MIN]))
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    for i in range(len(x)):
        x[i, MAX] = min(x[i, MAX], y[MAX])
        if x[i, MAX] < x[i, MIN]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
