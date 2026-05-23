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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


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


@njit(cache=True, fastmath=True)
def get_triggers_count_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_count_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\sum_i (x_i == a) = x_{n-1}`.

    :param domains: the domains of the variables, x is an alias for domains
    :type domains: NDArray
    :param parameters: the parameters of the propagator, a is the first parameter
    :type parameters: NDArray

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
    counter_min = counter[MIN]
    counter_max = counter[MAX]
    count_max = len(x)
    count_min = 0
    for x_i in x:
        x_i_min = x_i[MIN]
        x_i_max = x_i[MAX]
        if x_i_min > a or x_i_max < a:  # a is not in the domain: this x_i can never equal a
            count_max -= 1
            if count_max < counter_min:
                return PROP_INCONSISTENCY
        elif x_i_min == a and x_i_max == a:  # x_i is fixed to a
            count_min += 1
            if count_min > counter_max:
                return PROP_INCONSISTENCY
    if count_min > counter_min:
        counter[MIN] = count_min
    if count_max < counter_max:
        counter[MAX] = count_max
    if count_min == count_max:
        return PROP_ENTAILMENT
    if count_min == counter_max:  # we cannot have more domains equal to a
        all_different = True
        for x_i in x:
            x_i_min = x_i[MIN]
            x_i_max = x_i[MAX]
            if x_i_min == a:
                if x_i_max > a:
                    x_i[MIN] = a + 1
            elif x_i_min < a:
                if x_i_max == a:
                    x_i[MAX] = a - 1
                elif x_i_max > a:
                    all_different = False
        if all_different:
            return PROP_ENTAILMENT
    if count_max == counter_min:  # we cannot have more domains different from a
        for x_i in x:
            if x_i[MIN] <= a <= x_i[MAX]:
                x_i[:] = a
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
