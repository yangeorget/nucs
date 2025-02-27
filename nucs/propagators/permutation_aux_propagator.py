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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_permutation_aux(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n * n


@njit(cache=True)
def get_triggers_permutation_aux(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_permutation_aux(domains: NDArray, parameters: NDArray) -> int:
    """
    An auxiliary propagator needed to connect the next and prev variables of a permutation problem.
    :param domains: the domains of the variables, the next and prev variables
    :param parameters: the parameters of the propagator, unused here
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    n = len(domains) // 2
    next = domains[:n]
    prev = domains[n:]
    return (
        PROP_CONSISTENCY
        if filter_domains_permutation(n, next, prev) and filter_domains_permutation(n, prev, next)
        else PROP_INCONSISTENCY
    )


@njit(cache=True)
def filter_domains_permutation(n: int, next: NDArray, prev: NDArray) -> bool:
    for j in range(0, n):
        if prev[j, MIN] == prev[j, MAX]:
            i = prev[j, MIN]
            next[i, :] = j
        else:
            start = -1
            for i in range(prev[j, MIN], prev[j, MAX] + 1):
                if j < next[i, MIN] or j > next[i, MAX]:  # then prev[j] != i
                    if start == -1:
                        start = i
                    if i == prev[j, MIN]:
                        prev[j, MIN] += 1
                else:
                    start = -1
            if start >= 0:
                prev[j, MAX] = start - 1
                if prev[j, MAX] < prev[j, MIN]:
                    return False
    return True
