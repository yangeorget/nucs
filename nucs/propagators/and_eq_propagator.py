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


def get_complexity_and_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_and_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_and_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`\\& b_i = b_{n-1}` where for each i, b_i is a boolean variable.

    :param domains: the domains of the variables, b is an alias for domains
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[:-1]
    y = domains[-1]
    # A single scan replaces the former np.any / np.all reductions plus a separate candidate search:
    # it detects a false b_i (which forces the conjunction to 0), counts the undetermined b_i and
    # remembers the last one, so the cases below resolve from these two values alone.
    undetermined_nb = 0
    candidate_idx = -1
    for i in range(len(x)):
        if x[i, MAX] == 0:  # some b_i is 0, so the conjunction is 0
            y[MAX] = 0
            return PROP_INCONSISTENCY if y[MIN] > y[MAX] else PROP_ENTAILMENT
        if x[i, MIN] == 0:  # b_i is undetermined
            undetermined_nb += 1
            candidate_idx = i
    if undetermined_nb == 0:  # all b_i are 1, so the conjunction is 1
        y[MIN] = 1
        return PROP_INCONSISTENCY if y[MIN] > y[MAX] else PROP_ENTAILMENT
    if y[MIN] == 1:  # the conjunction is 1, so all b_i are 1
        x[:, MIN] = 1
        return PROP_ENTAILMENT  # no need to check for inconsistency
    if y[MAX] == 0 and undetermined_nb == 1:  # the conjunction is 0 with a single undetermined b_i
        x[candidate_idx, MAX] = 0
        return PROP_ENTAILMENT  # no need to check for inconsistency
    return PROP_CONSISTENCY
