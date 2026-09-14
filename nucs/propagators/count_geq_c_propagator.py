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
    LIVE_SET_MIN_ARITY,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_count_geq_c(n: int, parameters: NDArray) -> int:
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


STATE_LIVE_NB = 0  # trailed: how many x_i are still undetermined, biased by 1 so 0 reads as "cold"
STATE_COUNT_MIN = 1  # trailed: how many x_i are already fixed to a
STATE_REPORT = 2  # the engine's change-report cell, which must be the first cell of the hint suffix
STATE_LIVE = 3  # the live permutation: the undetermined x_i occupy its first STATE_LIVE_NB - 1 places


def get_state_count_geq_c(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: two trailed counters, the change-report cell, and a
    permutation of the x_i whose live prefix is the ones still undetermined.

    Laid out exactly like count_eq's, and for the same reason: an x_i that has lost a can never regain it
    and one that has ground onto a can never leave it, so the set worth looking at only shrinks down a
    branch. See get_state_count_eq for why only the size of that prefix is trailed and why count_max is
    derived rather than stored.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (2, 1 + n)
    :rtype: tuple[int, int]
    """
    if n < LIVE_SET_MIN_ARITY:
        return 0, 1
    return 2, 1 + n


@njit(cache=True)
def get_triggers_count_geq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


# inlined into the dispatch above: the narrow path is the one a short constraint takes on every
# call, and the call itself was measurable against it
@njit(cache=True, inline="always")
def _compute_domains_count_geq_c_plain(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Scans every variable: the plain path, for a propagator too narrow to carry a live set.

    Implements :math:`\\sum_i (x_i == a) >= c`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is the first parameter, c is the second parameter
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    a = int(parameters[0])
    c = int(parameters[1])
    count_max = len(domains)
    count_min = 0
    for domain in domains:
        domain_min = domain[DOMAIN_MIN]
        domain_max = domain[DOMAIN_MAX]
        if domain_min > a or domain_max < a:
            count_max -= 1
            if count_max < c:
                return PROP_INCONSISTENCY
        elif domain_min == a and domain_max == a:
            count_min += 1
            if count_min >= c:
                return PROP_ENTAILMENT
    if count_max == c:  # we cannot have more domains different from a
        for domain in domains:
            if domain[DOMAIN_MIN] <= a <= domain[DOMAIN_MAX]:
                domain[:] = a
        return PROP_ENTAILMENT
    # every write this propagator makes is in the branch above, which returns entailment, so reaching
    # here means nothing was written
    prop_state[0] = 0
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_count_geq_c(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`\\sum_i (x_i == a) >= c`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is the first parameter, c is the second parameter
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    if len(prop_state) < STATE_LIVE + len(domains):
        return _compute_domains_count_geq_c_plain(domains, parameters, prop_state)
    a = int(parameters[0])
    c = int(parameters[1])
    n = len(domains)
    live_nb = prop_state[STATE_LIVE_NB]
    if live_nb == 0:  # cold: a zeroed block, which is what the solver allocates and what a restart restores
        for i in range(n):
            prop_state[STATE_LIVE + i] = i
        live_nb = n
        count_min = 0
    else:
        live_nb -= 1  # stored biased by 1
        count_min = prop_state[STATE_COUNT_MIN]
    count_max = count_min + live_nb
    # the carried counts settle the constraint outright often enough to be worth testing before the scan:
    # count_max only falls down a branch and count_min only rises, so a carried count_max below c proves
    # the true one is too, and a carried count_min at or above c proves the same of the true one
    if count_max < c:
        return PROP_INCONSISTENCY
    if count_min >= c:
        return PROP_ENTAILMENT
    k = 0
    while k < live_nb:
        i = prop_state[STATE_LIVE + k]
        domain = domains[i]
        domain_min = domain[DOMAIN_MIN]
        domain_max = domain[DOMAIN_MAX]
        if domain_min > a or domain_max < a:
            count_max -= 1
            live_nb -= 1
            prop_state[STATE_LIVE + k] = prop_state[STATE_LIVE + live_nb]
            prop_state[STATE_LIVE + live_nb] = i
            if count_max < c:
                prop_state[STATE_LIVE_NB] = live_nb + 1
                prop_state[STATE_COUNT_MIN] = count_min
                return PROP_INCONSISTENCY
        elif domain_min == a and domain_max == a:
            count_min += 1
            live_nb -= 1
            prop_state[STATE_LIVE + k] = prop_state[STATE_LIVE + live_nb]
            prop_state[STATE_LIVE + live_nb] = i
            if count_min >= c:
                prop_state[STATE_LIVE_NB] = live_nb + 1
                prop_state[STATE_COUNT_MIN] = count_min
                return PROP_ENTAILMENT
        else:
            k += 1
    prop_state[STATE_LIVE_NB] = live_nb + 1
    prop_state[STATE_COUNT_MIN] = count_min
    if count_max == c:  # we cannot have more domains different from a
        for k in range(live_nb):
            domains[prop_state[STATE_LIVE + k]][:] = a
        return PROP_ENTAILMENT
    # every write this propagator makes is in the branch above, which returns entailment, so reaching
    # here means nothing was written
    prop_state[STATE_REPORT] = 0
    return PROP_CONSISTENCY
