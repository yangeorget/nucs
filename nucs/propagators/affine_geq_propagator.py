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

from nucs.constants import (
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_affine_geq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 2 * n


def get_triggers_affine_geq(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :param parameters: the parameters
    :return: an array of triggers
    """
    triggers = np.zeros(n, dtype=np.uint8)
    for i, c in enumerate(parameters[:-1]):
        if c < 0:
            triggers[i] = EVENT_MASK_MIN
        elif c > 0:
            triggers[i] = EVENT_MASK_MAX
    return triggers


@njit(cache=True)
def compute_domains_affine_geq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i >= a_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is an alias for parameters
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    domain_sum_min = domain_sum_max = parameters[-1]
    for i, c in enumerate(parameters[:-1]):
        if c > 0:
            domain_sum_min -= c * domains[i, MAX]
            domain_sum_max -= c * domains[i, MIN]
        elif c < 0:
            domain_sum_min -= c * domains[i, MIN]
            domain_sum_max -= c * domains[i, MAX]
    if domain_sum_max <= 0:
        return PROP_ENTAILMENT
    old_domains = np.copy(domains)
    for i, c in enumerate(parameters[:-1]):
        if c != 0:
            if c > 0:
                new_min = old_domains[i, MAX] - (domain_sum_min // -c)
                domains[i, MIN] = max(domains[i, MIN], new_min)
            else:
                new_max = old_domains[i, MIN] + (-domain_sum_min // -c)
                domains[i, MAX] = min(domains[i, MAX], new_max)
            if domains[i, MIN] > domains[i, MAX]:
                return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
