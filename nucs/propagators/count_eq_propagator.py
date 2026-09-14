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


def get_complexity_count_eq(n: int, parameters: NDArray) -> int:
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


def get_state_count_eq(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: two trailed counters, the change-report cell, and a
    permutation of the x_i whose live prefix is the ones still undetermined.

    An x_i is *undetermined* while a is inside its domain and it is not fixed to a; it leaves that state
    when a drops out of its domain (it can never equal a) or when it grounds on a (it must). Domains only
    narrow down a branch, so neither departure can be undone within one, and the propagator's whole first
    pass is over the ones that have not departed.

    Only the size of the live prefix is trailed, not the permutation, which is the sparse set's standard
    bargain: a departing x_i is swapped to the end of the prefix, so the places past the size hold exactly
    the ones that have left, and restoring the size re-admits exactly those. `count_max` is not stored at
    all -- with count_min fixed to a and live_nb able to go either way, `count_max = count_min + live_nb` --
    which keeps the trailed prefix the engine copies on every call down to two cells.

    :param n: the number of variables, one more than the number of x_i
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (2, 1 + (n - 1))
    :rtype: tuple[int, int]
    """
    if n < LIVE_SET_MIN_ARITY:
        return 0, 1
    return 2, n


@njit(cache=True)
def get_triggers_count_eq(n: int, variable: int, parameters: NDArray) -> int:
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
def _compute_domains_count_eq_plain(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Scans every variable: the plain path, for a propagator too narrow to carry a live set.

    Implements :math:`\\sum_i (x_i == a) = x_{n-1}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is the first parameter
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    a = int(parameters[0])
    x = domains[:-1]
    counter = domains[-1]
    # count_min = number of x_i already fixed to a, count_max = number that can still equal a;
    # the counter must lie in [count_min, count_max]. The counter bounds are read once into locals,
    # and the loop bails out as soon as count_max drops below them (too few possible) or count_min
    # rises above them (too many forced), saving the rest of the scan.
    counter_min = counter[DOMAIN_MIN]
    counter_max = counter[DOMAIN_MAX]
    count_max = len(x)
    count_min = 0
    for x_i in x:
        x_i_min = x_i[DOMAIN_MIN]
        x_i_max = x_i[DOMAIN_MAX]
        if x_i_min > a or x_i_max < a:  # a is not in the domain: this x_i can never equal a
            count_max -= 1
            if count_max < counter_min:
                return PROP_INCONSISTENCY
        elif x_i_min == a and x_i_max == a:  # x_i is fixed to a
            count_min += 1
            if count_min > counter_max:
                return PROP_INCONSISTENCY
    changed = False
    if count_min > counter_min:
        counter[DOMAIN_MIN] = count_min
        changed = True
    if count_max < counter_max:
        counter[DOMAIN_MAX] = count_max
        changed = True
    if count_min == count_max:
        return PROP_ENTAILMENT
    if count_min == counter_max:  # we cannot have more domains equal to a
        all_different = True
        for x_i in x:
            x_i_min = x_i[DOMAIN_MIN]
            x_i_max = x_i[DOMAIN_MAX]
            if x_i_min == a:
                if x_i_max > a:
                    x_i[DOMAIN_MIN] = a + 1
                    changed = True
            elif x_i_min < a:
                if x_i_max == a:
                    x_i[DOMAIN_MAX] = a - 1
                    changed = True
                elif x_i_max > a:
                    all_different = False
        if all_different:
            return PROP_ENTAILMENT
    if count_max == counter_min:  # we cannot have more domains different from a
        for x_i in x:
            if x_i[DOMAIN_MIN] <= a <= x_i[DOMAIN_MAX]:
                x_i[:] = a
        return PROP_ENTAILMENT
    if not changed:
        prop_state[0] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_count_eq(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`\\sum_i (x_i == a) = x_{n-1}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is the first parameter
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    if len(prop_state) < STATE_LIVE + len(domains) - 1:  # x is one shorter than domains
        return _compute_domains_count_eq_plain(domains, parameters, prop_state)
    a = int(parameters[0])
    x = domains[:-1]
    counter = domains[-1]
    n = len(x)
    # count_min = number of x_i already fixed to a, count_max = number that can still equal a;
    # the counter must lie in [count_min, count_max]. The counter bounds are read once into locals,
    # and the loop bails out as soon as count_max drops below them (too few possible) or count_min
    # rises above them (too many forced), saving the rest of the scan.
    counter_min = counter[DOMAIN_MIN]
    counter_max = counter[DOMAIN_MAX]
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
    # only the undetermined x_i are looked at: every other one has already been counted into count_min or
    # out of count_max, and cannot change which. Those that leave are swapped past the end of the live
    # prefix in the same pass, so the scan shrinks down the branch instead of restarting at n every call
    k = 0
    while k < live_nb:
        i = prop_state[STATE_LIVE + k]
        x_i = x[i]
        x_i_min = x_i[DOMAIN_MIN]
        x_i_max = x_i[DOMAIN_MAX]
        if x_i_min > a or x_i_max < a:  # a is not in the domain: this x_i can never equal a
            count_max -= 1
            live_nb -= 1
            prop_state[STATE_LIVE + k] = prop_state[STATE_LIVE + live_nb]
            prop_state[STATE_LIVE + live_nb] = i
            if count_max < counter_min:
                prop_state[STATE_LIVE_NB] = live_nb + 1
                prop_state[STATE_COUNT_MIN] = count_min
                return PROP_INCONSISTENCY
        elif x_i_min == a and x_i_max == a:  # x_i is fixed to a
            count_min += 1
            live_nb -= 1
            prop_state[STATE_LIVE + k] = prop_state[STATE_LIVE + live_nb]
            prop_state[STATE_LIVE + live_nb] = i
            if count_min > counter_max:
                prop_state[STATE_LIVE_NB] = live_nb + 1
                prop_state[STATE_COUNT_MIN] = count_min
                return PROP_INCONSISTENCY
        else:
            k += 1
    prop_state[STATE_LIVE_NB] = live_nb + 1
    prop_state[STATE_COUNT_MIN] = count_min
    changed = False
    if count_min > counter_min:
        counter[DOMAIN_MIN] = count_min
        changed = True
    if count_max < counter_max:
        counter[DOMAIN_MAX] = count_max
        changed = True
    if count_min == count_max:
        return PROP_ENTAILMENT
    if count_min == counter_max:  # we cannot have more domains equal to a
        all_different = True
        for k in range(live_nb):
            x_i = x[prop_state[STATE_LIVE + k]]
            if x_i[DOMAIN_MIN] == a:  # live, so its max is above a
                x_i[DOMAIN_MIN] = a + 1
                changed = True
            elif x_i[DOMAIN_MAX] == a:  # live, so its min is below a
                x_i[DOMAIN_MAX] = a - 1
                changed = True
            else:  # a is strictly inside, so this x_i cannot be pushed off it
                all_different = False
        if all_different:
            return PROP_ENTAILMENT
    if count_max == counter_min:  # we cannot have more domains different from a
        for k in range(live_nb):
            x[prop_state[STATE_LIVE + k]][:] = a
        return PROP_ENTAILMENT
    if not changed:
        prop_state[STATE_REPORT] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
