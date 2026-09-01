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


@njit(cache=True)
def smallest_minimal_value_var_heuristic(decision_variables: NDArray, domains: NDArray, params: NDArray) -> int:
    """
    Chooses the first variable which is not instantiated with the smallest minimal value.

    :param decision_variables: the decision variables
    :type decision_variables: NDArray
    :param domains: the domains
    :type domains: NDArray
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the variable
    :rtype: int
    """
    best_min = sys.maxsize
    best_variable = -1
    for variable in decision_variables:
        domain = domains[variable]
        if domain[DOMAIN_MIN] < domain[DOMAIN_MAX] and domain[DOMAIN_MIN] < best_min:  # not instantiated
            best_variable = variable
            best_min = domain[DOMAIN_MIN]
    return best_variable
