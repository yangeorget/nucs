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
from collections.abc import Sequence

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, PROP_CONSISTENCY, PROP_INCONSISTENCY


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


def get_state_scc(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    This propagator writes no domain, ever -- it is a feasibility check, answering only whether the digraph
    is still strongly connected. So it reports "nothing written" unconditionally, and the engine skips the
    write-back scan on every call rather than walking every variable to rediscover that. There is no cheaper
    case for the report to have, and no propagator that wants it more.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def compute_domains_scc(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Enforces that the digraph whose arcs are i -> j for j in [domains[i, DOMAIN_MIN], domains[i, DOMAIN_MAX]] is strongly connected.

    A digraph is strongly connected iff, from any single root (node 0), every node is reachable
    (forward DFS) and the root is reachable from every node (backward DFS). Both searches are
    iterative with an explicit stack: the former recursive dfs_row / dfs_col helpers could not use
    cache=True because of a Numba issue with cached recursion.

    The out-neighbors of every node form a contiguous interval, so the forward traversal never
    materializes the adjacency matrix and skips already-visited ranges via a union-find.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

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
        hi = domains[i, DOMAIN_MAX]
        j = next_unvisited(parent, domains[i, DOMAIN_MIN])
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
            if not visited[i] and domains[i, DOMAIN_MIN] <= c <= domains[i, DOMAIN_MAX]:
                visited[i] = True
                count += 1
                stack[sp] = i
                sp += 1
    if count != n:
        return PROP_INCONSISTENCY
    prop_state[0] = 0  # this propagator prunes nothing, so there is never anything to write back
    return PROP_CONSISTENCY
