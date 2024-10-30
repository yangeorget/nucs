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


def get_complexity_relation(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables, unused here
    :param parameters: the parameters
    :return: a float
    """
    return 3 * len(parameters)


def get_triggers_relation(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.ones((n, 2), dtype=np.bool)


@njit(cache=True)
def compute_domains_relation(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements a relation over n variables defined by its allowed tuples.
    :param domains: the domains of the variables
    :param parameters: the parameters of the propagator,
           the allowed tuples correspond to:
           (parameters_0, ..., parameters_n-1), (parameters_n, ..., parameters_2n-1), ...
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    n = len(domains)
    tuples = parameters.copy().reshape((-1, n))
    for domain_idx in range(n):
        tuples = tuples[
            (tuples[:, domain_idx] >= domains[domain_idx, MIN]) & (tuples[:, domain_idx] <= domains[domain_idx, MAX])
        ]
        if len(tuples) == 0:
            return PROP_INCONSISTENCY
    for domain_idx in range(n):  # no support for .min(axis=0) in Numba
        domains[domain_idx, MIN] = tuples[:, domain_idx].min()
        domains[domain_idx, MAX] = tuples[:, domain_idx].max()
    if len(tuples) == 1:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
