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


def get_complexity_count_leq_c(n: int, parameters: NDArray) -> int:
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


def get_state_count_leq_c(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


@njit(cache=True)
def get_triggers_count_leq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_count_leq_c(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`S\\sum_i (x_i == a) <= c`.

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
            if count_max <= c:
                return PROP_ENTAILMENT
        elif domain_min == a and domain_max == a:
            count_min += 1
            if count_min > c:
                return PROP_INCONSISTENCY
    changed = False
    if count_min == c:  # we cannot have more domains equal to a
        all_different = True
        for domain in domains:
            domain_min = domain[DOMAIN_MIN]
            domain_max = domain[DOMAIN_MAX]
            if domain_min == a:
                if domain_max > a:
                    domain[DOMAIN_MIN] = a + 1
                    changed = True
            elif domain_min < a:
                if domain_max == a:
                    domain[DOMAIN_MAX] = a - 1
                    changed = True
                elif domain_max > a:
                    all_different = False
        if all_different:
            return PROP_ENTAILMENT
    if not changed:
        prop_state[0] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
