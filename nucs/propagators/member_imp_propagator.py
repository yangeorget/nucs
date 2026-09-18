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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_member_imp(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the allowed values
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return len(parameters)


@njit(cache=True)
def get_triggers_member_imp(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param variable: the index of the variable
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    # b (variable 0) only needs to wake when it becomes true (b_min rises to 1): b becoming false makes the
    # implication vacuous, so there is nothing to deduce. x still needs both bounds.
    return EVENT_MASK_MIN if variable == 0 else EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_member_imp(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements the half-reified (implied) constraint :math:`b \\rightarrow x \\in \\{a_0, ..., a_{n-1}\\}`.

    Unlike the fully-reified ``member_reif``, this only snaps x's bounds onto the allowed values when b is true
    and applies the contrapositive (b becomes false when no allowed value is left in x's range); it never forces
    b true when x is always in the set nor pushes x off the allowed values when b is false.

    :param domains: the domains of the variables, b is the first domain, x is the second domain
    :type domains: NDArray
    :param parameters: the allowed values, in strictly ascending order
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    b = domains[0]
    x = domains[1]
    if b[DOMAIN_MAX] == 0:  # b is false: the implication is vacuously satisfied
        return PROP_ENTAILMENT
    n = len(parameters)
    # The allowed values still in x's range are parameters[lo..hi]; the window is empty when lo > hi.
    lo = 0
    while lo < n and parameters[lo] < x[DOMAIN_MIN]:
        lo += 1
    hi = n - 1
    while hi >= 0 and parameters[hi] > x[DOMAIN_MAX]:
        hi -= 1
    if lo > hi:  # no allowed value is left in range, so b must be false
        if b[DOMAIN_MIN] == 1:
            return PROP_INCONSISTENCY
        b[:] = 0
        return PROP_ENTAILMENT
    if b[DOMAIN_MIN] == 1:  # x is in the set: snap the bounds onto the window's ends
        x[DOMAIN_MIN] = parameters[lo]
        x[DOMAIN_MAX] = parameters[hi]
    # As many allowed values as x has values: x is always in the set.
    if hi - lo == x[DOMAIN_MAX] - x[DOMAIN_MIN]:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
