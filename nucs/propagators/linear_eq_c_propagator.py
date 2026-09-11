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
    EVENT_MASK_NONE,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_linear_eq_c(n: int, parameters: NDArray) -> int:
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


def get_state_linear_eq_c(n: int, parameters: Sequence[int]) -> tuple[int, int]:
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
def get_triggers_linear_eq_c(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX if parameters[variable] != 0 else EVENT_MASK_NONE


@njit(cache=True)
def compute_domains_linear_eq_c(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements :math:`\\sum_i a_i * x_i = a_{n}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is an alias for parameters
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    factors = parameters[:-1]
    n = len(factors)
    # domain_sum_min / domain_sum_max bracket the value of (sum a_i * x_i - a_n): domain_sum_min is
    # its largest possible value, domain_sum_max its smallest (the swapped naming is shared with the
    # geq/leq propagators). The bounds are recomputed and re-filtered until a fixpoint is reached.
    domain_sum_min = domain_sum_max = -parameters[-1]
    for i in range(n):
        factor = factors[i]
        x_min = domains[i, DOMAIN_MIN]
        x_max = domains[i, DOMAIN_MAX]
        if factor > 0:
            domain_sum_min += factor * x_max
            domain_sum_max += factor * x_min
        else:
            domain_sum_min += factor * x_min
            domain_sum_max += factor * x_max
    # If the sum is forced strictly above or below a_n the equality is unsatisfiable. This single
    # global test catches every inconsistency, so the per-variable x[DOMAIN_MIN] > x[DOMAIN_MAX] check that used
    # to sit inside the filtering loop below is redundant and was dropped.
    if domain_sum_max > 0 or domain_sum_min < 0:
        return PROP_INCONSISTENCY
    # The two bounds meet exactly when every variable is bound or has a zero factor, which is what the
    # loop above used to count: subtracting them term by term leaves sum |a_i| * (x_max - x_min), a sum of
    # non-negative terms, each zero only for a variable that cannot move the total. Reading entailment off
    # the two accumulators instead takes a compare and an add per variable out of the scan -- and with them
    # a data-dependent branch, in the one loop here that is otherwise straight-line arithmetic.
    if domain_sum_min == domain_sum_max:
        return PROP_ENTAILMENT
    changed = False
    for i in range(n):
        factor = factors[i]
        if factor == 0:
            continue
        x_min = domains[i, DOMAIN_MIN]
        x_max = domains[i, DOMAIN_MAX]
        if x_min == x_max:
            continue
        if factor > 0:
            new_min = x_max - (domain_sum_min // factor)
            new_max = x_min + (-domain_sum_max // factor)
        else:
            new_min = x_max - (domain_sum_max // factor)
            new_max = x_min + (-domain_sum_min // factor)
        if new_min > x_min:
            domains[i, DOMAIN_MIN] = new_min
            changed = True
        if new_max < x_max:
            domains[i, DOMAIN_MAX] = new_max
            changed = True
    if not changed:
        prop_state[0] = 0  # nothing written: the engine can skip the write-back scan
    return PROP_CONSISTENCY
