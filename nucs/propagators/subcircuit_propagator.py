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

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, PROP_CONSISTENCY, PROP_INCONSISTENCY

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


@njit(cache=True)
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


@njit(cache=True)
def compute_domains_subcircuit(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces that the successor array forms a sub-circuit: the nodes i with x_i != i form a single circuit
    while the remaining nodes are self-loops (x_i = i, excluded). The empty sub-circuit (all self-loops) is
    allowed. This is a self-loop-aware variant of the no-sub-cycle constraint and is meant to run alongside
    an alldifferent on the same variables.

    The i-th variable is the successor of node i and takes the label of a node, which is ``offset + i``: the
    successors are 0-based by default and the offset makes any other contiguous node numbering (a 1-based
    MiniZinc array, say) usable without shifting every variable into an auxiliary one. Successors are first
    trimmed to the node labels, which is what bounds the node indices this propagator derives from them.

    :param domains: the domains of the variables (the successors)
    :type domains: NDArray
    :param parameters: the node label offset, parameters[0], or no parameter at all for 0-based successors
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    offset = int(parameters[0]) if len(parameters) > 0 else 0
    for i in range(n):
        domains[i, DOMAIN_MIN] = max(domains[i, DOMAIN_MIN], offset)
        domains[i, DOMAIN_MAX] = min(domains[i, DOMAIN_MAX], offset + n - 1)
        if domains[i, DOMAIN_MIN] > domains[i, DOMAIN_MAX]:
            return PROP_INCONSISTENCY
    # committed[i] is True when node i is necessarily active (part of the single circuit), i.e. it cannot be
    # an excluded self-loop: either i is no longer in its own domain, or a fixed active arc touches it
    # (its source, and -- because x is a permutation -- its target).
    committed = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if domains[i, DOMAIN_MIN] > i + offset or domains[i, DOMAIN_MAX] < i + offset:
            committed[i] = True
        if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX] and domains[i, DOMAIN_MIN] != i + offset:
            committed[i] = True
            committed[domains[i, DOMAIN_MIN] - offset] = True
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
            if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
                j = domains[i, DOMAIN_MIN] - offset
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
                        if domains[end, DOMAIN_MIN] == start + offset:
                            domains[end, DOMAIN_MIN] += 1
                        if domains[end, DOMAIN_MAX] == start + offset:
                            domains[end, DOMAIN_MAX] -= 1
                        if domains[end, DOMAIN_MIN] > domains[end, DOMAIN_MAX]:
                            return PROP_INCONSISTENCY
                        if end < i:
                            loop = True
    return PROP_CONSISTENCY
