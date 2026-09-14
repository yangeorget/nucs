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


STATE_REPORT = 0  # the engine's change-report cell, which must be the first cell of the hint suffix
STATE_STAMP = 1  # which call the reachability marks below belong to
STATE_REACH = 2  # the forward layer marks, then the backward ones, (length + 1) * (q_nb + 1) of each
# then the supported-symbol bounds, one pair per position, collected by the backward pass itself


def get_state_regular(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the change-report cell, a call stamp, and the two
    layered-graph reachability tables this propagator used to allocate with np.zeros on every call.

    The tables are marked rather than cleared. Each call takes the next stamp and writes it where it used
    to write a 1; a cell holding any other stamp reads as unreachable, so the O(length * q) clearing a
    fresh np.zeros used to provide for free is not needed at all -- which is the larger half of what the
    allocation was costing. The whole block is an untrailed hint: a stale mark is one that does not match
    the current stamp, so staleness is what makes the scheme work rather than something to restore.

    :param n: the number of variables in the sequence
    :type n: int
    :param parameters: the automaton, whose first entry is the number of states
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 2 + 2 * (n + 1) * (Q + 1) + 2n)
    :rtype: tuple[int, int]
    """
    q_nb = int(parameters[0])
    return 0, 2 + 2 * (n + 1) * (q_nb + 1) + 2 * n


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


def is_vacuous_regular(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    """
    Returns whether the automaton and the initial domains make the constraint vacuous.

    An automaton whose transition function is total and whose every state accepts reads any word over its
    alphabet into an accepting state, so it rejects nothing. That alone is not enough: the propagator also
    keeps each variable inside the alphabet, dropping any value outside 1..S, so it still filters when a
    domain reaches beyond it. The domains are therefore required to lie within the alphabet already -- which
    holds for the whole search, since domains only shrink.

    Accepting states are required of every state rather than of the reachable ones only, which is sufficient
    and costs a single pass.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the DFA description
    :type parameters: Sequence[int]
    :param domains: the initial domains of the sequence variables, as (min, max) pairs
    :type domains: Sequence[tuple[int, int]]

    :return: True when no assignment can violate the constraint
    :rtype: bool
    """
    q_nb = parameters[0]
    s_nb = parameters[1]
    q0 = parameters[2]
    if q0 < 1 or q0 > q_nb:  # an unusable initial state is not something to reason about here
        return False
    for domain_min, domain_max in domains:
        if domain_min < 1 or domain_max > s_nb:
            return False
    for transition in range(q_nb * s_nb):
        if parameters[3 + transition] == 0:  # a missing transition rejects that symbol
            return False
    acc_off = 3 + q_nb * s_nb
    for q in range(q_nb):
        if not parameters[acc_off + q]:
            return False
    return True


@njit(cache=True)
def compute_domains_regular(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
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
    can be pruned, which is exact for a binary alphabet (no interior value to remove).

    :param domains: the domains of the sequence variables
    :type domains: NDArray
    :param parameters: the DFA description, as above
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

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
    row = q_nb + 1
    fwd_off = STATE_REACH
    bwd_off = STATE_REACH + (length + 1) * row
    sup_min_off = STATE_REACH + 2 * (length + 1) * row
    sup_max_off = sup_min_off + length
    stamp = prop_state[STATE_STAMP] + 1
    if stamp <= 0:  # the stamp has run out of int32: clear once and start again
        prop_state[STATE_REACH:] = 0
        stamp = 1
    prop_state[STATE_STAMP] = stamp
    # forward reachability
    prop_state[fwd_off + q0] = stamp
    for i in range(length):
        var = domains[i]
        fwd_base = fwd_off + i * row
        next_base = fwd_base + row
        for q in range(1, q_nb + 1):
            if prop_state[fwd_base + q] == stamp:
                for v in range(max(1, var[DOMAIN_MIN]), min(s_nb, var[DOMAIN_MAX]) + 1):
                    nq = parameters[3 + (q - 1) * s_nb + (v - 1)]
                    if nq != 0:
                        prop_state[next_base + nq] = stamp
    # backward reachability, and the supported symbols in the same sweep. A symbol v is supported at
    # position i exactly when some forward-reachable q reads it into a state that still accepts -- which is
    # the pair (q, v) this loop is already visiting, so collecting the support bounds here costs only the
    # early exit, and only for the states that are forward-reachable. It replaces a separate pass that
    # re-derived the same thing per candidate bound, and that pass was 64% of this propagator's model.
    last_base = bwd_off + length * row
    for q in range(1, q_nb + 1):
        if parameters[acc_off + (q - 1)]:
            prop_state[last_base + q] = stamp
    for i in range(length - 1, -1, -1):
        var = domains[i]
        lo = max(1, var[DOMAIN_MIN])
        hi = min(s_nb, var[DOMAIN_MAX])
        bwd_base = bwd_off + i * row
        next_base = bwd_base + row
        fwd_base = fwd_off + i * row
        sup_min = s_nb + 1
        sup_max = 0
        for q in range(1, q_nb + 1):
            forward = prop_state[fwd_base + q] == stamp
            marked = False
            for v in range(lo, hi + 1):
                nq = parameters[3 + (q - 1) * s_nb + (v - 1)]
                if nq != 0 and prop_state[next_base + nq] == stamp:
                    if not marked:
                        prop_state[bwd_base + q] = stamp
                        marked = True
                    if not forward:
                        break  # q cannot start a path here, so it supports nothing; bwd is all it gives
                    sup_min = min(sup_min, v)
                    sup_max = max(sup_max, v)
        prop_state[sup_min_off + i] = sup_min
        prop_state[sup_max_off + i] = sup_max
    if prop_state[bwd_off + q0] != stamp:
        return PROP_INCONSISTENCY  # the initial state cannot reach acceptance
    # prune each variable's bounds to the supported symbols the sweep above recorded
    changed = False
    for i in range(length):
        new_min = prop_state[sup_min_off + i]
        new_max = prop_state[sup_max_off + i]
        if new_min > new_max:
            return PROP_INCONSISTENCY
        var = domains[i]
        if new_min != var[DOMAIN_MIN] or new_max != var[DOMAIN_MAX]:
            var[DOMAIN_MIN] = new_min
            var[DOMAIN_MAX] = new_max
            changed = True
    ground_nb = 0
    for i in range(length):
        if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
            ground_nb += 1
    if ground_nb == length:
        return PROP_ENTAILMENT  # a single accepted word remains
    if not changed:
        prop_state[STATE_REPORT] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
