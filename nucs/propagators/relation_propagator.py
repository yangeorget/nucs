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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def compute_domains_relation(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements a relation over n variables defined by its allowed tuples.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: the parameters of the propagator,
           the allowed tuples correspond to:
           (parameters_0, ..., parameters_n-1), (parameters_n, ..., parameters_2n-1), ...
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains)
    tuple_nb = len(parameters) // n
    # Single allocation-free pass over the tuples: a tuple is valid when every value lies within the current
    # domain bounds. We accumulate, per column, the min and max over the valid tuples in a small scratch array
    # (we cannot write the result into domains yet, since the bounds are still needed to test validity). This
    # avoids copying the whole tuple table -- and allocating a fresh array per filtered column -- on every call.
    bounds = np.empty((n, 2), dtype=domains.dtype)
    valid_nb = 0
    offset = 0
    for _ in range(tuple_nb):
        valid = True
        for col in range(n):
            value = parameters[offset + col]
            if value < domains[col, MIN] or value > domains[col, MAX]:
                valid = False
                break
        if valid:
            if valid_nb == 0:
                for col in range(n):
                    value = parameters[offset + col]
                    bounds[col, MIN] = value
                    bounds[col, MAX] = value
            else:
                for col in range(n):
                    value = parameters[offset + col]
                    if value < bounds[col, MIN]:
                        bounds[col, MIN] = value
                    if value > bounds[col, MAX]:
                        bounds[col, MAX] = value
            valid_nb += 1
        offset += n
    if valid_nb == 0:
        return PROP_INCONSISTENCY
    for col in range(n):
        domains[col, MIN] = bounds[col, MIN]
        domains[col, MAX] = bounds[col, MAX]
    if valid_nb == 1:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
