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

from nucs.constants import EVENT_MASK_GROUND, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY

PATH_START = 0
PATH_END = 1
PATH_LENGTH = 2


def get_complexity_no_sub_cycle(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n * n


def get_triggers_no_sub_cycle(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.full(n, dtype=np.uint8, fill_value=EVENT_MASK_GROUND)


@njit(cache=True)
def compute_domains_no_sub_cycle(domains: NDArray, parameters: NDArray) -> int:
    """
    :param domains: the domains of the variables
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    n = len(domains)
    paths = np.zeros((n, 3), dtype=np.int16)
    for i in range(n):
        paths[i, :PATH_LENGTH] = i
    loop = True
    while loop:
        loop = False
        for i in range(n):
            if domains[i, MIN] == domains[i, MAX]:
                j = domains[i, MIN]
                if i == j:
                    return PROP_INCONSISTENCY
                if paths[i, PATH_END] == i:
                    end = paths[i, PATH_END] = paths[j, PATH_END]
                    start = paths[j, PATH_START] = paths[i, PATH_START]
                    paths[start, PATH_END] = end
                    paths[end, PATH_START] = start
                    length = paths[i, PATH_LENGTH] + 1 + paths[j, PATH_LENGTH]
                    paths[i, PATH_LENGTH] = paths[j, PATH_LENGTH] = paths[start, PATH_LENGTH] = paths[
                        end, PATH_LENGTH
                    ] = length
                    if length < n - 1:
                        if domains[end, MIN] == start:
                            domains[end, MIN] = start + 1
                        if domains[end, MAX] == start:
                            domains[end, MAX] = start - 1
                        if domains[end, MIN] > domains[end, MAX]:
                            return PROP_INCONSISTENCY
                        if end < i:
                            loop = True
    return PROP_CONSISTENCY
