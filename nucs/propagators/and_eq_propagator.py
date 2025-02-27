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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_and_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


@njit(cache=True)
def get_triggers_and_eq(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_and_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements & b_i = b_{n-1} where for each i, b_i is a boolean variable.
    :param domains: the domains of the variables, b is an alias for domains
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    x = domains[:-1]
    y = domains[-1]
    if np.any(x[:, MAX] == 0):
        y[MAX] = 0
        return PROP_INCONSISTENCY if y[MIN] > y[MAX] else PROP_ENTAILMENT
    if np.all(x[:, MIN] == 1):
        y[MIN] = 1
        return PROP_INCONSISTENCY if y[MIN] > y[MAX] else PROP_ENTAILMENT
    if y[MIN] == 1:
        x[:, MIN] = 1
        return PROP_ENTAILMENT  # no need to check for inconsistency
    elif y[MAX] == 0:
        candidates_nb = 0
        candidate_idx = -1
        for i in range(len(x)):
            if x[i, MIN] == 0:
                candidate_idx = i
                candidates_nb += 1
        if candidates_nb == 1:
            x[candidate_idx, MAX] = 0
            return PROP_ENTAILMENT  # no need to check for inconsistency
    return PROP_CONSISTENCY
