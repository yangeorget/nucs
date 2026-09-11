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


def get_complexity_sum_eq(n: int, parameters: NDArray) -> int:
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


def get_state_sum_eq(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    The cell is untrailed, and could not be anything else: it describes the call that has just happened,
    not the node, so there is nothing about it to restore. The engine pre-sets it to 1 and reads it back
    once, between the call and the write-back it decides.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


@njit(cache=True)
def get_triggers_sum_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_sum_eq(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`\\sum_i x_i = x_{n-1}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) - 1
    y_min = domains[-1, DOMAIN_MIN]
    y_max = domains[-1, DOMAIN_MAX]
    domain_sum_min = -y_min
    domain_sum_max = -y_max
    unbound_count = 0 if y_min == y_max else 1
    for i in range(n):
        x_min = domains[i, DOMAIN_MIN]
        x_max = domains[i, DOMAIN_MAX]
        domain_sum_min += x_max
        domain_sum_max += x_min
        if x_min < x_max:
            unbound_count += 1
    if unbound_count == 0:
        return PROP_ENTAILMENT if domain_sum_min == 0 else PROP_INCONSISTENCY
    changed = False
    for i in range(n):
        x_min = domains[i, DOMAIN_MIN]
        x_max = domains[i, DOMAIN_MAX]
        if x_min == x_max:
            continue
        new_min = x_max - domain_sum_min
        new_max = x_min - domain_sum_max
        if new_min > x_min:
            domains[i, DOMAIN_MIN] = new_min
            changed = True
        if new_max < x_max:
            domains[i, DOMAIN_MAX] = new_max
            changed = True
        if domains[i, DOMAIN_MIN] > domains[i, DOMAIN_MAX]:
            return PROP_INCONSISTENCY
    if y_min < y_max:
        new_min = y_max + domain_sum_max
        new_max = y_min + domain_sum_min
        if new_min > y_min:
            domains[-1, DOMAIN_MIN] = new_min
            changed = True
        if new_max < y_max:
            domains[-1, DOMAIN_MAX] = new_max
            changed = True
        if domains[-1, DOMAIN_MIN] > domains[-1, DOMAIN_MAX]:
            return PROP_INCONSISTENCY
    if unbound_count == 1:
        return PROP_ENTAILMENT
    if not changed:
        prop_state[0] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
