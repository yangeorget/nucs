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

STATE_REPORT = 0  # the engine's change-report cell


def value_precede_chain_parameters(chain: Sequence[int]) -> list[int]:
    """
    Returns the parameters of the propagator for a chain of distinct values: the chain length k, the smallest
    chain value, the k chain values, then the chain index of every value from the smallest to the largest chain
    value (-1 for a value outside the chain).

    :param chain: the values c[0], c[1], ..., in the order their first occurrences must follow
    :type chain: Sequence[int]

    :return: the parameters
    :rtype: list[int]
    """
    base = min(chain)
    table = [-1] * (max(chain) - base + 1)
    for index, value in enumerate(chain):
        table[value - base] = index
    return [len(chain), base, *chain, *table]


def get_complexity_value_precede_chain(n: int, parameters: NDArray) -> int:
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


def get_state_value_precede_chain(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    The cell is untrailed: it describes the call that has just happened, not the node.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


@njit(cache=True)
def get_triggers_value_precede_chain(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX  # any bound change may lower how far the chain can be realised


@njit(cache=True)
def compute_domains_value_precede_chain(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements value_precede_chain: the first occurrences of c[0], c[1], ..., c[k-1] in x come in that order
    (a value may not occur at all, but then no later chain value occurs either). Values outside the chain are free.

    One forward pass keeps r, the highest chain index such that c[0], ..., c[r] can occur in order in the prefix
    before position i (a greedy subsequence match, which is optimal). Position i then cannot take c[m] for any
    m >= r + 2, which prunes its bounds; r grows by at most one per position, so the pass stops as soon as r
    reaches k - 1. Pruning position i only affects later positions, so one pass is a fixpoint.

    :param domains: the domains of the variables (the array x)
    :type domains: NDArray
    :param parameters: the parameters, as built by value_precede_chain_parameters
    :type parameters: NDArray
    :param prop_state: this propagator's state block, the change-report cell
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    k = parameters[0]
    base = parameters[1]
    table_start = 2 + k
    table_size = len(parameters) - table_start
    changed = False
    fixed = True  # whether every position so far is fixed
    r = -1
    for i in range(len(domains)):
        lo = domains[i, DOMAIN_MIN]
        hi = domains[i, DOMAIN_MAX]
        forbidden = r + 2  # the lowest chain index position i cannot take
        new_lo = lo
        while new_lo <= hi:
            j = new_lo - base
            if 0 <= j < table_size and parameters[table_start + j] >= forbidden:
                new_lo += 1
            else:
                break
        new_hi = hi
        while new_hi >= new_lo:
            j = new_hi - base
            if 0 <= j < table_size and parameters[table_start + j] >= forbidden:
                new_hi -= 1
            else:
                break
        if new_lo > new_hi:
            return PROP_INCONSISTENCY
        if new_lo != lo:
            domains[i, DOMAIN_MIN] = new_lo
            changed = True
        if new_hi != hi:
            domains[i, DOMAIN_MAX] = new_hi
            changed = True
        if new_lo <= parameters[2 + r + 1] <= new_hi:
            r += 1
        fixed = fixed and new_lo == new_hi
        if r == k - 1:
            # nothing is forbidden past here; with a fixed prefix, every chain value has already occurred in order
            if fixed:
                return PROP_ENTAILMENT
            break
    if not changed:
        prop_state[STATE_REPORT] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
