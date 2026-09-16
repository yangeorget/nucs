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


def get_complexity_circuit_positions(n: int, parameters: NDArray) -> int:
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


def get_state_circuit_positions(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: scratch space for the two breadth-first searches and the
    fixed chains, which compute_domains_circuit_positions would otherwise allocate on every call.

    Every cell is fully overwritten before it is read, so the whole block is untrailed scratch.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 4n): forward distances, backward distances, a queue, fixed predecessors
    :rtype: tuple[int, int]
    """
    return 0, 4 * n


@njit(cache=True)
def get_triggers_circuit_positions(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator: the position windows depend on every bound, not only on fixed
    successors.

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
def compute_domains_circuit_positions(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Enforces that the successors form a single circuit, reasoning about the position of each node on the tour.

    The i-th variable is the successor of node i and takes the label ``offset + j`` of a node j. Node 0 opens the
    tour at position 0, so node j sits at a position p with dist(0, j) <= p <= n - dist(j, 0), the distances
    being taken over the edges the successor bounds still allow. Then:

    - a node whose predecessor is fixed can be entered from that predecessor only, so no other edge into it counts;
    - a node that node 0 cannot reach, or that cannot reach node 0, makes the circuit impossible; this is also how a
      closed sub-cycle is detected;
    - j is removed from a bound of succ[i] when j's position window cannot follow i's, and node 0 when i cannot be
      last;
    - the end of a fixed chain is kept from closing it back onto its start, as NO_SUB_CYCLE does.

    These are the bounds the position variables of MiniZinc's standard circuit decomposition would reach, without
    the variables. Pruning an edge changes the distances, so the rules are iterated to a fixpoint, which makes the
    propagator idempotent.

    :param domains: the domains of the successors
    :type domains: NDArray
    :param parameters: the node label offset, parameters[0]
    :type parameters: NDArray
    :param prop_state: this propagator's state block: scratch for distances, a queue and fixed predecessors
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    offset = int(parameters[0])
    fwd = prop_state[0:n]
    bwd = prop_state[n : 2 * n]
    queue = prop_state[2 * n : 3 * n]
    fixed_pred = prop_state[3 * n : 4 * n]  # 1 + the fixed predecessor of each node, or 0
    for i in range(n):
        domains[i, DOMAIN_MIN] = max(domains[i, DOMAIN_MIN], offset)
        domains[i, DOMAIN_MAX] = min(domains[i, DOMAIN_MAX], offset + n - 1)
        if n > 1:  # a single node is its own successor; otherwise a self-loop is a sub-cycle
            if domains[i, DOMAIN_MIN] == offset + i:
                domains[i, DOMAIN_MIN] += 1
            if domains[i, DOMAIN_MAX] == offset + i:
                domains[i, DOMAIN_MAX] -= 1
        if domains[i, DOMAIN_MIN] > domains[i, DOMAIN_MAX]:
            return PROP_INCONSISTENCY
    if n == 1:
        return PROP_ENTAILMENT
    while True:
        changed = False
        # the end of a fixed chain cannot close it back onto its start unless the chain spans every node
        for i in range(n):
            fixed_pred[i] = 0
        for i in range(n):
            if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
                j = domains[i, DOMAIN_MIN] - offset
                if fixed_pred[j] != 0:
                    return PROP_INCONSISTENCY  # two nodes fixed to the same successor
                fixed_pred[j] = i + 1
        for start in range(n):
            if fixed_pred[start] != 0 or domains[start, DOMAIN_MIN] != domains[start, DOMAIN_MAX]:
                continue
            end = start
            length = 0
            while domains[end, DOMAIN_MIN] == domains[end, DOMAIN_MAX] and length < n:
                end = domains[end, DOMAIN_MIN] - offset
                length += 1
            if length < n - 1:
                if domains[end, DOMAIN_MIN] == start + offset:
                    domains[end, DOMAIN_MIN] += 1
                    changed = True
                if domains[end, DOMAIN_MAX] == start + offset:
                    domains[end, DOMAIN_MAX] -= 1
                    changed = True
                if domains[end, DOMAIN_MIN] > domains[end, DOMAIN_MAX]:
                    return PROP_INCONSISTENCY
        # forward distances from node 0
        for i in range(n):
            fwd[i] = -1
            bwd[i] = -1
        fwd[0] = 0
        queue[0] = 0
        head = 0
        tail = 1
        while head < tail:
            i = queue[head]
            head += 1
            for label in range(domains[i, DOMAIN_MIN], domains[i, DOMAIN_MAX] + 1):
                j = label - offset
                if fwd[j] < 0 and (fixed_pred[j] == 0 or fixed_pred[j] == i + 1):
                    fwd[j] = fwd[i] + 1
                    queue[tail] = j
                    tail += 1
        if tail < n:
            return PROP_INCONSISTENCY  # some node cannot be reached from node 0
        # backward distances to node 0
        bwd[0] = 0
        queue[0] = 0
        head = 0
        tail = 1
        while head < tail:
            j = queue[head]
            head += 1
            for i in range(n):
                if (
                    bwd[i] < 0
                    and domains[i, DOMAIN_MIN] <= offset + j <= domains[i, DOMAIN_MAX]
                    and (fixed_pred[j] == 0 or fixed_pred[j] == i + 1)
                ):
                    bwd[i] = bwd[j] + 1
                    queue[tail] = i
                    tail += 1
        if tail < n:
            return PROP_INCONSISTENCY  # some node cannot reach node 0
        # position windows: [fwd[j], n - bwd[j]] for j > 0, and [0, 0] for node 0
        for j in range(1, n):
            if fwd[j] > n - bwd[j]:
                return PROP_INCONSISTENCY
        all_fixed = True
        for i in range(n):
            i_min = fwd[i]
            i_max = 0 if i == 0 else n - bwd[i]
            lo = domains[i, DOMAIN_MIN]
            hi = domains[i, DOMAIN_MAX]
            # a bound moved by this loop may land on the node itself, so the self-loop is excluded here too
            while lo <= hi and not _allowed(i, lo - offset, i_min, i_max, fwd, bwd, fixed_pred, n):
                lo += 1
            while lo <= hi and not _allowed(i, hi - offset, i_min, i_max, fwd, bwd, fixed_pred, n):
                hi -= 1
            if lo > hi:
                return PROP_INCONSISTENCY
            if lo != domains[i, DOMAIN_MIN] or hi != domains[i, DOMAIN_MAX]:
                domains[i, DOMAIN_MIN] = lo
                domains[i, DOMAIN_MAX] = hi
                changed = True
            if lo != hi:
                all_fixed = False
        if not changed:
            # every node reachable both ways over fixed successors is a single circuit
            return PROP_ENTAILMENT if all_fixed else PROP_CONSISTENCY


@njit(cache=True)
def _allowed(i: int, j: int, i_min: int, i_max: int, fwd: NDArray, bwd: NDArray, fixed_pred: NDArray, n: int) -> bool:
    """
    Returns whether node j can be the successor of node i, whose position lies in [i_min, i_max].

    A bound moved by the pruning loop may land on i itself, and a node whose predecessor is fixed can be entered
    from that predecessor only, so both are excluded here as well as by the position windows.

    :param i: the node
    :type i: int
    :param j: the candidate successor
    :type j: int
    :param i_min: the smallest position of the predecessor
    :type i_min: int
    :param i_max: the largest position of the predecessor
    :type i_max: int
    :param fwd: the distances from node 0
    :type fwd: NDArray
    :param bwd: the distances to node 0
    :type bwd: NDArray
    :param fixed_pred: 1 + the fixed predecessor of each node, or 0
    :type fixed_pred: NDArray
    :param n: the number of nodes
    :type n: int

    :return: True when the two position windows allow it
    :rtype: bool
    """
    if j == i or (fixed_pred[j] != 0 and fixed_pred[j] != i + 1):
        return False
    if j == 0:
        return i_max == n - 1  # closing the tour: the predecessor must be able to be last
    return max(fwd[j], i_min + 1) <= min(n - bwd[j], i_max + 1)
