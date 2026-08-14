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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_regular(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables (the sequence length)
    :type n: int
    :param parameters: the DFA description, starting with the state count and the symbol count
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    q = int(parameters[0])
    s = int(parameters[1])
    return n * q * s


@njit(cache=True)
def get_triggers_regular(n: int, variable: int, parameters: NDArray) -> int:
    """
    Triggered whenever a bound of a sequence variable changes.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index, unused here
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def _supported(domains: NDArray, parameters: NDArray, fwd: NDArray, bwd: NDArray, i: int, v: int, q_nb: int, s_nb: int) -> bool:
    """
    Returns whether symbol ``v`` at position ``i`` lies on a valid path (a forward-reachable state reads it into
    a state that can still reach acceptance).
    """
    if v < 1 or v > s_nb:
        return False
    for q in range(1, q_nb + 1):
        if fwd[i, q]:
            nq = parameters[3 + (q - 1) * s_nb + (v - 1)]
            if nq != 0 and bwd[i + 1, nq]:
                return True
    return False


@njit(cache=True)
def compute_domains_regular(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the regular constraint: the sequence of variables must be accepted by a deterministic finite
    automaton.

    ``parameters = [Q, S, q0, d_0, ..., d_{Q*S-1}, a_0, ..., a_{Q-1}]`` where Q is the number of states
    (numbered 1..Q), S the number of symbols (the values 1..S), q0 the initial state, ``d[(q-1)*S + (v-1)]``
    the state reached from state q on symbol v (0 meaning no transition), and ``a[q-1]`` whether state q is
    accepting.

    Filtering follows Pesant's layered graph: a forward pass computes the states reachable at each position and
    a backward pass the states from which acceptance is still reachable; a symbol is kept only when some
    forward-reachable state reads it into a state that can still accept. On the interval domains only a bound
    can be pruned, which is exact for a binary alphabet (no interior value to remove). The passes are iterated
    to a fixpoint so a single call is idempotent.

    :param domains: the domains of the sequence variables
    :type domains: NDArray
    :param parameters: the DFA description, as above
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    length = len(domains)
    q_nb = parameters[0]
    s_nb = parameters[1]
    q0 = parameters[2]
    acc_off = 3 + q_nb * s_nb
    if length == 0:
        return PROP_ENTAILMENT if parameters[acc_off + (q0 - 1)] else PROP_INCONSISTENCY
    fwd = np.zeros((length + 1, q_nb + 1), dtype=np.uint8)
    bwd = np.zeros((length + 1, q_nb + 1), dtype=np.uint8)
    change = True
    while change:
        change = False
        # forward reachability
        fwd[:] = 0
        fwd[0, q0] = 1
        for i in range(length):
            var = domains[i]
            for q in range(1, q_nb + 1):
                if fwd[i, q]:
                    for v in range(max(1, var[MIN]), min(s_nb, var[MAX]) + 1):
                        nq = parameters[3 + (q - 1) * s_nb + (v - 1)]
                        if nq != 0:
                            fwd[i + 1, nq] = 1
        # backward reachability
        bwd[:] = 0
        for q in range(1, q_nb + 1):
            if parameters[acc_off + (q - 1)]:
                bwd[length, q] = 1
        for i in range(length - 1, -1, -1):
            var = domains[i]
            for q in range(1, q_nb + 1):
                for v in range(max(1, var[MIN]), min(s_nb, var[MAX]) + 1):
                    nq = parameters[3 + (q - 1) * s_nb + (v - 1)]
                    if nq != 0 and bwd[i + 1, nq]:
                        bwd[i, q] = 1
                        break
        if not bwd[0, q0]:
            return PROP_INCONSISTENCY  # the initial state cannot reach acceptance
        # prune each variable's bounds to the supported symbols
        for i in range(length):
            var = domains[i]
            new_min = var[MIN]
            while new_min <= var[MAX] and not _supported(domains, parameters, fwd, bwd, i, new_min, q_nb, s_nb):
                new_min += 1
            new_max = var[MAX]
            while new_max >= new_min and not _supported(domains, parameters, fwd, bwd, i, new_max, q_nb, s_nb):
                new_max -= 1
            if new_min > new_max:
                return PROP_INCONSISTENCY
            if new_min != var[MIN] or new_max != var[MAX]:
                var[MIN] = new_min
                var[MAX] = new_max
                change = True
    ground_nb = 0
    for i in range(length):
        if domains[i, MIN] == domains[i, MAX]:
            ground_nb += 1
    if ground_nb == length:
        return PROP_ENTAILMENT  # a single accepted word remains
    return PROP_CONSISTENCY
