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
from nucs.numpy import new_triggers
from nucs.propagators.affine_eq_propagator import compute_domain_sum_max, compute_domain_sum_min


def get_complexity_affine_geq(n: int, parameters: NDArray) -> float:
    return 5 * n


def get_triggers_affine_geq(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = new_triggers(n, False)
    for i, c in enumerate(parameters[:-1]):
        triggers[i, MIN] = c < 0
        triggers[i, MAX] = c > 0
    return triggers


@njit(cache=True)
def compute_domains_affine_geq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i >= a_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is an alias for parameters
    """
    if compute_domain_sum_max(domains, parameters) <= 0:
        return PROP_ENTAILMENT
    domain_sum_min = compute_domain_sum_min(domains, parameters)
    new_domains = np.copy(domains)
    for i, c in enumerate(parameters[:-1]):
        if c != 0:
            if c > 0:
                new_min = domains[i, MAX] - (domain_sum_min // -c)
                new_domains[i, MIN] = max(domains[i, MIN], new_min)
            else:
                new_max = domains[i, MIN] + (-domain_sum_min // -c)
                new_domains[i, MAX] = min(domains[i, MAX], new_max)
                if new_domains[i, MIN] > new_domains[i, MAX]:
                    return PROP_INCONSISTENCY
    domains[:] = new_domains[:]  # numba does not support copyto
    return PROP_CONSISTENCY
