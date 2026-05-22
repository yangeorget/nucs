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


def get_complexity_scc(n: int, parameters: NDArray) -> int:
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
def get_triggers_scc(n: int, variable: int, parameters: NDArray) -> int:
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
def next_unvisited(parent: NDArray, i: int) -> int:
    """
    Returns the smallest unvisited index greater than or equal to i, using path compression.

    A visited index v points to v + 1; the sentinel parent[n] = n marks "none".

    :param parent: the union-find array of next-unvisited pointers
    :type parent: NDArray
    :param i: an index
    :type i: int

    :return: the smallest unvisited index >= i
    :rtype: int
    """
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


@njit(cache=True, fastmath=True)
def compute_domains_scc(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces that the digraph whose arcs are i -> j for j in [domains[i, MIN], domains[i, MAX]] is strongly connected.

    The out-neighbors of every node form a contiguous interval, so the forward traversal never
    materializes the adjacency matrix and skips already-visited ranges via a union-find.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    stack = np.empty(n, dtype=np.int32)
    # forward DFS from node 0: every node must be reachable
    parent = np.arange(n + 1, dtype=np.int32)  # next-unvisited union-find
    parent[0] = 1
    count = 1
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        i = stack[sp]
        hi = domains[i, MAX]
        j = next_unvisited(parent, domains[i, MIN])
        while j <= hi:
            parent[j] = j + 1
            count += 1
            stack[sp] = j
            sp += 1
            j = next_unvisited(parent, j)
    if count != n:
        return PROP_INCONSISTENCY
    # backward DFS from node 0: node 0 must be reachable from every node
    visited = np.zeros(n, dtype=np.bool_)
    visited[0] = True
    count = 1
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        c = stack[sp]
        for i in range(n):
            if not visited[i] and domains[i, MIN] <= c <= domains[i, MAX]:
                visited[i] = True
                count += 1
                stack[sp] = i
                sp += 1
    if count != n:
        return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
