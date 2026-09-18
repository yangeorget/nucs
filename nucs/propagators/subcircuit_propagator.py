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

ON_CYCLE = 2  # the visited mark of the nodes of a fixed cycle


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
    # A call is O(n) per pass, but ranking it as the O(n) propagators would queue it ahead of the element and
    # reified propagators that narrow the successors, and wake it again after each of them: on a prize-collecting
    # TSP, n here costs 2.8-3.4x the calls of n * n for the same tree, and 7-16% of the search rate.
    return n * n


def get_state_subcircuit(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: scratch space for the fixed predecessors, the chain starts
    and the nodes the chain walks have visited, which compute_domains_subcircuit would otherwise allocate on every
    call.

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
def get_triggers_subcircuit(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator: a bound can land on a label the fixed successors rule out, or leave a
    node's own label, without any successor becoming fixed, so every bound change counts.

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
def compute_domains_subcircuit(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Enforces that the successor array forms a sub-circuit, in O(n), from what the fixed successors rule out: the
    nodes i with x_i != i form a single circuit while the remaining nodes are self-loops (x_i = i, excluded). The
    empty sub-circuit (all self-loops) is allowed. It is meant to run alongside an alldifferent on the same
    variables.

    The i-th variable is the successor of node i and takes the label ``offset + j`` of a node j: the successors are
    0-based by default and the offset makes any other contiguous node numbering (a 1-based MiniZinc array, say)
    usable without shifting every variable into an auxiliary one. A node is committed when it cannot be a
    self-loop: its own label is out of its domain, or another node is fixed to it. A successor bound is moved past
    a label that cannot follow node i:

    - a node whose predecessor is fixed to another node, or to itself as a self-loop, which also takes i's own
      label from a committed i;
    - when i ends a fixed chain and a committed node lies outside it, the chain's start, which would close it.

    A fixed cycle fixes every other node to a self-loop, and fails the propagator if one of them is committed. Only
    a newly fixed successor or a newly committed node changes what the rules rule out, so they are repeated until
    a pass makes neither, which makes the propagator idempotent.

    :param domains: the domains of the successors
    :type domains: NDArray
    :param parameters: the node label offset, parameters[0], or no parameter at all for 0-based successors
    :type parameters: NDArray
    :param prop_state: this propagator's state block: scratch for fixed predecessors, chain starts and visited nodes
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    offset = int(parameters[0]) if len(parameters) > 0 else 0
    fixed_pred = prop_state[0:n]  # 1 + the fixed predecessor of each node, itself for a self-loop, or 0
    chain_start = prop_state[n : 2 * n]  # 1 + the start a fixed chain's end cannot close onto, or 0
    visited = prop_state[2 * n : 3 * n]
    for i in range(n):
        domains[i, DOMAIN_MIN] = max(domains[i, DOMAIN_MIN], offset)
        domains[i, DOMAIN_MAX] = min(domains[i, DOMAIN_MAX], offset + n - 1)
        if domains[i, DOMAIN_MIN] > domains[i, DOMAIN_MAX]:
            return PROP_INCONSISTENCY
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
                if fixed_pred[j] != 0:
                    return PROP_INCONSISTENCY  # two nodes fixed to the same successor
                fixed_pred[j] = i + 1
        committed_nb = 0
        for i in range(n):
            if _is_committed(domains, i, offset, fixed_pred):
                committed_nb += 1
        # walk each fixed chain from its start: its end cannot close it while a committed node lies outside it
        for start in range(n):
            if fixed_pred[start] != 0 or domains[start, DOMAIN_MIN] != domains[start, DOMAIN_MAX]:
                continue  # a self-loop is its own predecessor, so it starts no chain
            visited[start] = 1
            end = start
            length = 0
            while domains[end, DOMAIN_MIN] == domains[end, DOMAIN_MAX]:
                end = domains[end, DOMAIN_MIN] - offset
                visited[end] = 1
                length += 1
            if length + 1 < committed_nb:  # every node of the chain is committed
                chain_start[end] = start + 1
        # a fixed node no chain walk reached, other than a self-loop, lies on a fixed cycle: the sub-circuit
        for i in range(n):
            if visited[i] == 0 and domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX] and fixed_pred[i] != i + 1:
                j = i
                while visited[j] == 0:
                    visited[j] = ON_CYCLE
                    j = domains[j, DOMAIN_MIN] - offset
                for k in range(n):
                    if visited[k] != ON_CYCLE:
                        if domains[k, DOMAIN_MIN] > k + offset or domains[k, DOMAIN_MAX] < k + offset:
                            return PROP_INCONSISTENCY  # a committed node off the sub-circuit
                        domains[k, DOMAIN_MIN] = domains[k, DOMAIN_MAX] = k + offset
                return PROP_ENTAILMENT
        if fixed_nb == n:
            return PROP_ENTAILMENT  # every node a self-loop
        changed = False
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
            if lo == hi or (lo > i + offset and domains[i, DOMAIN_MIN] <= i + offset):
                changed = True  # newly fixed, or newly committed
            elif hi < i + offset and domains[i, DOMAIN_MAX] >= i + offset:
                changed = True  # newly committed
            domains[i, DOMAIN_MIN] = lo
            domains[i, DOMAIN_MAX] = hi
        if not changed:
            return PROP_CONSISTENCY


@njit(cache=True)
def _is_committed(domains: NDArray, i: int, offset: int, fixed_pred: NDArray) -> bool:
    """
    Returns whether node i cannot be a self-loop: its own label is out of its domain, or another node is fixed to it.

    :param domains: the domains of the successors
    :type domains: NDArray
    :param i: the node
    :type i: int
    :param offset: the node label offset
    :type offset: int
    :param fixed_pred: 1 + the fixed predecessor of each node, itself for a self-loop, or 0
    :type fixed_pred: NDArray

    :return: True when node i is committed
    :rtype: bool
    """
    if domains[i, DOMAIN_MIN] > i + offset or domains[i, DOMAIN_MAX] < i + offset:
        return True
    return fixed_pred[i] != 0 and fixed_pred[i] != i + 1


@njit(cache=True)
def _excluded(i: int, j: int, fixed_pred: NDArray, chain_start: NDArray) -> bool:
    """
    Returns whether the fixed successors rule out node j as the successor of node i.

    :param i: the node
    :type i: int
    :param j: the candidate successor
    :type j: int
    :param fixed_pred: 1 + the fixed predecessor of each node, itself for a self-loop, or 0
    :type fixed_pred: NDArray
    :param chain_start: 1 + the start each fixed chain's end cannot close onto, or 0
    :type chain_start: NDArray

    :return: True when j cannot follow i
    :rtype: bool
    """
    return (fixed_pred[j] != 0 and fixed_pred[j] != i + 1) or chain_start[i] == j + 1
