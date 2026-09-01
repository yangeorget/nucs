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
import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DECISION_EQ, DOMAIN_MAX, DOMAIN_MIN


@njit(cache=True)
def min_cost_dom_heuristic(domains: NDArray, variable: int, params: NDArray) -> tuple[int, int]:
    """
    Chooses the value that minimizes the cost.

    When no value in the domain has a positive cost the choice falls back to the min of the domain: the
    split has to partition the domain for the enumeration to stay complete, and an out-of-domain value
    would not.

    :param domains: the domains
    :type domains: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional (first dimension corresponds to variables, second to values) cost array
    :type params: NDArray

    :return: the kind of the decision and the value the domain is split at
    :rtype: Tuple[int, int]
    """
    best_cost = sys.maxsize
    domain = domains[variable]
    best_value = domain[DOMAIN_MIN]
    for value in range(domain[DOMAIN_MIN], domain[DOMAIN_MAX] + 1):
        cost = params[variable][value]
        if 0 < cost < best_cost:
            best_cost = cost
            best_value = value
    return DECISION_EQ, best_value
