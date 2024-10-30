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

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_affine_eq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 5 * n


def get_triggers_affine_eq(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.ones((n, 2), dtype=np.bool)


@njit(cache=True)
def compute_domain_sum_min(domains: NDArray, parameters: NDArray) -> int:
    domain_sum_min = parameters[-1]
    for i, c in enumerate(parameters[:-1]):
        domain_sum_min -= c * (domains[i, MAX] if c > 0 else domains[i, MIN])
    return domain_sum_min


@njit(cache=True)
def compute_domain_sum_max(domains: NDArray, parameters: NDArray) -> int:
    domain_sum_max = parameters[-1]
    for i, c in enumerate(parameters[:-1]):
        domain_sum_max -= c * (domains[i, MIN] if c > 0 else domains[i, MAX])
    return domain_sum_max


@njit(cache=True)
def compute_domains_affine_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i = a_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is an alias for parameters
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    domain_sum_min = compute_domain_sum_min(domains, parameters)
    domain_sum_max = compute_domain_sum_max(domains, parameters)
    new_domains = np.copy(domains)
    for i, c in enumerate(parameters[:-1]):
        if c != 0:
            if c > 0:
                new_min = domains[i, MAX] - (domain_sum_min // -c)
                new_max = domains[i, MIN] + (domain_sum_max // c)
            else:
                new_min = domains[i, MAX] - (-domain_sum_max // c)
                new_max = domains[i, MIN] + (-domain_sum_min // -c)
            new_domains[i, MIN] = max(domains[i, MIN], new_min)
            new_domains[i, MAX] = min(domains[i, MAX], new_max)
            if new_domains[i, MIN] > new_domains[i, MAX]:
                return PROP_INCONSISTENCY
    domains[:] = new_domains[:]  # numba does not support copyto
    return PROP_CONSISTENCY
