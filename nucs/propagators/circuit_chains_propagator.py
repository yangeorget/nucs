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

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_circuit_chains(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n


def get_state_circuit_chains(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: scratch space for the fixed predecessors, the chain starts
    and the nodes the chain walks have visited, which compute_domains_circuit_chains would otherwise allocate on
    every call.

    Every cell is fully overwritten before it is read, so the whole block is untrailed scratch.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 3n)
    :rtype: tuple[int, int]
    """
    return 0, 3 * n


@njit(cache=True)
def get_triggers_circuit_chains(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator: a bound can land on a label the fixed successors rule out without any
    successor becoming fixed, so every bound change counts.

    :param n: the number of variables
    :type n: int
    :param variable: the variable, unused here
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_circuit_chains(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Enforces that the successors form a single circuit, in O(n), from what the fixed successors rule out.

    The i-th variable is the successor of node i and takes the label ``offset + j`` of a node j. A successor bound is
    moved past a label that cannot follow node i:

    - i itself, a self-loop being a sub-cycle;
    - a node whose predecessor is fixed to another node;
    - when i ends a fixed chain spanning fewer than all the nodes, the chain's start, which would close it.

    A fixed cycle over fewer than all the nodes fails the propagator. Only a newly fixed successor changes what the
    rules rule out, so they are repeated until a pass fixes none, which makes the propagator idempotent.

    It wakes on any bound change, not only on a fixed successor: a chain's start or a node with
    a fixed predecessor can become a bound without anything being fixed. Bounding each node's position on the tour
    by its distance from node 0, as MiniZinc's order-variable decomposition does, was measured to prune beyond these
    rules in at most 0.5% of calls on real inputs, for several times the cost, and is not done.

    :param domains: the domains of the successors
    :type domains: NDArray
    :param parameters: the node label offset, parameters[0]
    :type parameters: NDArray
    :param prop_state: this propagator's state block: scratch for fixed predecessors, chain starts and visited nodes
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    offset = int(parameters[0])
    fixed_pred = prop_state[0:n]  # 1 + the fixed predecessor of each node, or 0
    chain_start = prop_state[n : 2 * n]  # 1 + the start a fixed chain's end cannot close onto, or 0
    visited = prop_state[2 * n : 3 * n]
    for i in range(n):
        domains[i, DOMAIN_MIN] = max(domains[i, DOMAIN_MIN], offset)
        domains[i, DOMAIN_MAX] = min(domains[i, DOMAIN_MAX], offset + n - 1)
        if domains[i, DOMAIN_MIN] > domains[i, DOMAIN_MAX]:
            return PROP_INCONSISTENCY
    if n == 1:
        return PROP_ENTAILMENT
    while True:
        for i in range(n):
            fixed_pred[i] = 0
            chain_start[i] = 0
            visited[i] = 0
        fixed_nb = 0
        for i in range(n):
            if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
                fixed_nb += 1
                j = domains[i, DOMAIN_MIN] - offset
                if j == i or fixed_pred[j] != 0:
                    return PROP_INCONSISTENCY  # a self-loop, or two nodes fixed to the same successor
                fixed_pred[j] = i + 1
        # walk each fixed chain from its start: its end cannot close it unless it spans every node
        for start in range(n):
            if fixed_pred[start] != 0 or domains[start, DOMAIN_MIN] != domains[start, DOMAIN_MAX]:
                continue
            visited[start] = 1
            end = start
            length = 0
            while domains[end, DOMAIN_MIN] == domains[end, DOMAIN_MAX]:
                end = domains[end, DOMAIN_MIN] - offset
                visited[end] = 1
                length += 1
            if length < n - 1:
                chain_start[end] = start + 1
        # a fixed node no chain walk reached lies on a fixed cycle, which must span every node
        for i in range(n):
            if visited[i] == 0 and domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
                j = i
                length = 0
                while True:
                    visited[j] = 1
                    j = domains[j, DOMAIN_MIN] - offset
                    length += 1
                    if j == i:
                        break
                if length < n:
                    return PROP_INCONSISTENCY
                return PROP_ENTAILMENT  # a fixed circuit over every node
        newly_fixed = False
        for i in range(n):
            lo = domains[i, DOMAIN_MIN]
            hi = domains[i, DOMAIN_MAX]
            if lo == hi:
                continue
            while lo <= hi and _excluded(i, lo - offset, fixed_pred, chain_start):
                lo += 1
            while lo <= hi and _excluded(i, hi - offset, fixed_pred, chain_start):
                hi -= 1
            if lo > hi:
                return PROP_INCONSISTENCY
            domains[i, DOMAIN_MIN] = lo
            domains[i, DOMAIN_MAX] = hi
            if lo == hi:
                newly_fixed = True
        if not newly_fixed:
            return PROP_CONSISTENCY


@njit(cache=True)
def _excluded(i: int, j: int, fixed_pred: NDArray, chain_start: NDArray) -> bool:
    """
    Returns whether the fixed successors rule out node j as the successor of node i.

    :param i: the node
    :type i: int
    :param j: the candidate successor
    :type j: int
    :param fixed_pred: 1 + the fixed predecessor of each node, or 0
    :type fixed_pred: NDArray
    :param chain_start: 1 + the start each fixed chain's end cannot close onto, or 0
    :type chain_start: NDArray

    :return: True when j cannot follow i
    :rtype: bool
    """
    return j == i or (fixed_pred[j] != 0 and fixed_pred[j] != i + 1) or chain_start[i] == j + 1
