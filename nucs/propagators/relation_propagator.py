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

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_relation(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return len(parameters)


# The state block: the number of tuples ruled out (trailed), the change report, then the live-tuple
# permutation (both hints).
STATE_DEAD_NB = 0  # trailed
STATE_REPORT = 1  # the first hint cell, which is where the engine looks
STATE_INITIALIZED = 2  # whether the permutation below has been built
STATE_LIVE = 3


def get_state_relation(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: a live-tuple sparse set and the change report.

    A tuple that no longer fits inside the domains can never fit again, because domains only narrow, so the
    set of tuples still worth testing shrinks monotonically down a branch. Keeping it means the scan costs
    the tuples still alive rather than every tuple in the table, which is simple tabular reduction and what
    Gecode's and Choco's table propagators do.

    Only the *count* of ruled-out tuples is trailed; the permutation itself is a hint. Removing a tuple swaps
    it past the end of the live prefix, and every swap stays inside the window the choice point handed down,
    so restoring the count restores the *set* however a sibling branch reordered it. A backtrack therefore
    revives exactly the tuples it should, without the permutation being saved.

    Unlike an active-variable prefix over a linear constraint -- which measured slower, because indirecting
    to skip a multiply-add costs more than the multiply-add -- skipping a tuple here skips a loop over the
    columns, so the indirection is amortised over the width of the table.

    :param n: the number of variables
    :type n: int
    :param parameters: the allowed tuples, flattened
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb)
    :rtype: tuple[int, int]
    """
    tuple_nb = len(parameters) // n if n else 0
    return 1, 2 + tuple_nb


@njit(cache=True)
def get_triggers_relation(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_relation(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements a relation over n variables defined by its allowed tuples.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: the parameters of the propagator,
           the allowed tuples correspond to:
           (parameters_0, ..., parameters_n-1), (parameters_n, ..., parameters_2n-1), ...
    :type parameters: NDArray
    :param prop_state: this propagator's state block: the ruled-out count, the change report and the
        live-tuple permutation
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    tuple_nb = len(parameters) // n
    live = prop_state[STATE_LIVE:]
    if prop_state[STATE_INITIALIZED] == 0:
        # cold: a zeroed block is not a permutation, and the report cell cannot stand in for this flag
        # because the engine pre-sets it to 1 before every call. The flag is a hint rather than trailed
        # because what it guards stays a permutation for good: only the live count says how much is alive.
        prop_state[STATE_INITIALIZED] = 1
        for t in range(tuple_nb):
            live[t] = t
    live_nb = tuple_nb - prop_state[STATE_DEAD_NB]
    # One pass over the tuples still alive. A tuple is valid when every value lies within the current domain
    # bounds; one that is not is swapped out of the live prefix, since narrowing can never bring it back.
    # We accumulate, per column, the min and max over the valid tuples in a small scratch array (we cannot
    # write the result into domains yet, since the bounds are still needed to test validity).
    bounds = np.empty((n, 2), dtype=domains.dtype)
    valid_nb = 0
    position = 0
    while position < live_nb:
        offset = live[position] * n
        valid = True
        for col in range(n):
            value = parameters[offset + col]
            if value < domains[col, DOMAIN_MIN] or value > domains[col, DOMAIN_MAX]:
                valid = False
                break
        if not valid:
            live_nb -= 1
            swapped = live[live_nb]
            live[live_nb] = live[position]
            live[position] = swapped  # the departing tuple is kept, so the prefix stays a permutation
            continue  # the tuple swapped into this position has not been tested yet
        if valid_nb == 0:
            for col in range(n):
                value = parameters[offset + col]
                bounds[col, DOMAIN_MIN] = value
                bounds[col, DOMAIN_MAX] = value
        else:
            for col in range(n):
                value = parameters[offset + col]
                bounds[col, DOMAIN_MIN] = min(bounds[col, DOMAIN_MIN], value)
                bounds[col, DOMAIN_MAX] = max(bounds[col, DOMAIN_MAX], value)
        valid_nb += 1
        position += 1
    prop_state[STATE_DEAD_NB] = tuple_nb - live_nb
    if valid_nb == 0:
        return PROP_INCONSISTENCY
    changed = False
    for col in range(n):
        if bounds[col, DOMAIN_MIN] != domains[col, DOMAIN_MIN] or bounds[col, DOMAIN_MAX] != domains[col, DOMAIN_MAX]:
            domains[col, DOMAIN_MIN] = bounds[col, DOMAIN_MIN]
            domains[col, DOMAIN_MAX] = bounds[col, DOMAIN_MAX]
            changed = True
    if valid_nb == 1:
        return PROP_ENTAILMENT
    if not changed:
        prop_state[STATE_REPORT] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
