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
# Copyright 2024 - Yan Georget
###############################################################################
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_min_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 3 * n


def get_triggers_min_eq(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.full(n, dtype=np.uint8, fill_value=EVENT_MASK_MIN_MAX)


@njit(cache=True)
def compute_domains_min_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Min_i x_i = x_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    x = domains[:-1]
    y = domains[-1]
    y[MIN] = max(y[MIN], np.min(x[:, MIN]))
    y[MAX] = min(y[MAX], np.min(x[:, MAX]))
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    candidates_nb = 0
    candidate_idx = -1
    for i in range(len(x)):
        if x[i, MIN] <= y[MIN]:
            x[i, MIN] = y[MIN]
            candidate_idx = i
            candidates_nb += 1
    if candidates_nb == 1:
        x[candidate_idx, MAX] = y[MAX]
    return PROP_CONSISTENCY
