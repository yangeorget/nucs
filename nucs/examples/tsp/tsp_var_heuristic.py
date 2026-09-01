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

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.heuristics.max_regret_var_heuristic import regret


@njit(cache=True)
def tsp_var_heuristic(decision_variables: NDArray, domains: NDArray, params: NDArray) -> int:
    """
    :param decision_variables: the decision variables
    :type decision_variables: NDArray
    :param domains: the domains
    :type domains: NDArray
    :param params: a two-dimensional (first dimension correspond to variables, second to values) cost array
    :type params: NDArray
    :return: the variable
    :rtype: int
    """
    best_score = -sys.maxsize
    best_variable = -1
    for variable in decision_variables:
        domain = domains[variable]
        if 0 < domain[DOMAIN_MAX] - domain[DOMAIN_MIN]:
            score = compute_score(domain, variable, params)
            if best_score < score:
                best_variable = variable
                best_score = score
    return best_variable


@njit(cache=True)
def compute_score(domain: NDArray, variable: int, params: NDArray) -> int:
    """
    Minimize [min(12, size(X)), -regret(X)] for lexicographic order.
    """
    size = min(12, domain[DOMAIN_MAX] - domain[DOMAIN_MIN] + 1)
    return -size * 1024 + regret(domain, params[variable])
