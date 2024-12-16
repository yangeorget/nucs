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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_permutation_aux(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n


def get_triggers_permutation_aux(n: int, parameters: NDArray) -> NDArray:
    """
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    triggers = np.zeros(n, dtype=np.uint8)
    # triggers[:-1] = EVENT_MASK_MIN_MAX  # not worth the cost
    triggers[-1] = EVENT_MASK_MIN_MAX
    return triggers


@njit(cache=True)
def compute_domains_permutation_aux(domains: NDArray, parameters: NDArray) -> int:
    """
    :param domains: the domains of the variables
    :param parameters: the parameters of the propagator
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    next = domains[:-1]
    n = len(next)
    prev_j = domains[-1]
    j = parameters[0]
    for i in range(0, n):
        if i < prev_j[MIN] or i > prev_j[MAX]:  # if prev_j != i
            # then next_i != j
            if next[i, MIN] == j:
                next[i, MIN] += 1
            if next[i, MAX] == j:
                next[i, MAX] -= 1
            if next[i, MIN] > next[i, MAX]:
                return PROP_INCONSISTENCY
    if prev_j[MIN] == prev_j[MAX]:
        next[prev_j[MIN]] = j
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
