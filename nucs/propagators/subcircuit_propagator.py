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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY

START = 0
END = 1
LENGTH = 2


def get_complexity_subcircuit(n: int, parameters: NDArray) -> int:
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


@njit(cache=True, fastmath=True)
def get_triggers_subcircuit(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_subcircuit(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces that the 0-based successor array forms a sub-circuit: the nodes i with x_i != i form a single
    circuit while the remaining nodes are self-loops (x_i = i, excluded). The empty sub-circuit (all
    self-loops) is allowed. This is a self-loop-aware variant of the no-sub-cycle constraint and is meant to
    run alongside an alldifferent on the same variables.

    :param domains: the domains of the variables (0-based successors)
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    # committed[i] is True when node i is necessarily active (part of the single circuit), i.e. it cannot be
    # an excluded self-loop: either i is no longer in its own domain, or a fixed active arc touches it
    # (its source, and -- because x is a permutation -- its target).
    committed = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if domains[i, MIN] > i or domains[i, MAX] < i:
            committed[i] = True
        if domains[i, MIN] == domains[i, MAX] and domains[i, MIN] != i:
            committed[i] = True
            committed[domains[i, MIN]] = True
    total_committed = 0
    for i in range(n):
        if committed[i]:
            total_committed += 1
    # Build chains from fixed active arcs (self-loops are excluded nodes and skipped). A chain may close into
    # a circuit only once it contains every committed node; otherwise closing it would leave an active node
    # in a separate component, which a sub-circuit forbids.
    paths = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        paths[i, START] = i
        paths[i, END] = i
    loop = True
    while loop:
        loop = False
        for i in range(n):
            if domains[i, MIN] == domains[i, MAX]:
                j = domains[i, MIN]
                if i == j:  # excluded self-loop: not part of any chain
                    continue
                if paths[i, END] == i:
                    end = paths[i, END] = paths[j, END]
                    start = paths[j, START] = paths[i, START]
                    paths[start, END] = end
                    paths[end, START] = start
                    length = paths[i, LENGTH] + 1 + paths[j, LENGTH]
                    paths[i, LENGTH] = paths[j, LENGTH] = paths[start, LENGTH] = paths[end, LENGTH] = length
                    if length + 1 < total_committed:  # a committed node remains outside this chain
                        if domains[end, MIN] == start:
                            domains[end, MIN] = start + 1
                        if domains[end, MAX] == start:
                            domains[end, MAX] = start - 1
                        if domains[end, MIN] > domains[end, MAX]:
                            return PROP_INCONSISTENCY
                        if end < i:
                            loop = True
    return PROP_CONSISTENCY
