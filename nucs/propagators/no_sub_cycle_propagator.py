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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_GROUND, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY

PATH_START = 0
PATH_END = 1
PATH_LENGTH = 2


def get_complexity_no_sub_cycle(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n


@njit(cache=True)
def get_triggers_no_sub_cycle(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_GROUND


@njit(cache=True)
def compute_domains_no_sub_cycle(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces that a permutation does not contain any sub-cycle.

    The i-th variable is the successor of node i and takes the label of a node, which is ``offset + i``: the
    successors are 0-based by default and the offset makes any other contiguous node numbering (a 1-based
    MiniZinc array, say) usable without shifting every variable into an auxiliary one. Successors are first
    trimmed to the node labels, which is what bounds the node indices this propagator derives from them.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: the node label offset, parameters[0], or no parameter at all for 0-based successors
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    offset = int(parameters[0]) if len(parameters) > 0 else 0
    for i in range(n):
        domains[i, MIN] = max(domains[i, MIN], offset)
        domains[i, MAX] = min(domains[i, MAX], offset + n - 1)
        if domains[i, MIN] > domains[i, MAX]:
            return PROP_INCONSISTENCY
    paths = np.zeros((n, 3), dtype=np.int16)
    for i in range(n):
        paths[i, :PATH_LENGTH] = i
    loop = True
    while loop:
        loop = False
        for i in range(n):
            if domains[i, MIN] == domains[i, MAX]:
                j = domains[i, MIN] - offset
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
                        # closing the chain back onto its start would make a sub-cycle: forbid that label
                        if domains[end, MIN] == start + offset:
                            domains[end, MIN] = start + offset + 1
                        if domains[end, MAX] == start + offset:
                            domains[end, MAX] = start + offset - 1
                        if domains[end, MIN] > domains[end, MAX]:
                            return PROP_INCONSISTENCY
                        if end < i:
                            loop = True
    return PROP_CONSISTENCY
