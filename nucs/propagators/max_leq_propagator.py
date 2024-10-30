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
# Copyright 2024 - Yan Georget
###############################################################################
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_max_leq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 3 * n


def get_triggers_max_leq(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    triggers = np.zeros((n, 2), dtype=np.bool)
    for i in range(n - 1):
        triggers[i, MIN] = True
    triggers[-1, MAX] = True
    return triggers


@njit(cache=True)
def compute_domains_max_leq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Max_i x_i <= x_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    x = domains[:-1]
    y = domains[-1]
    if np.max(x[:, MAX]) <= y[MIN]:
        return PROP_ENTAILMENT
    y[MIN] = max(y[MIN], np.max(x[:, MIN]))
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    for i in range(len(x)):
        x[i, MAX] = min(x[i, MAX], y[MAX])
        if x[i, MAX] < x[i, MIN]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
