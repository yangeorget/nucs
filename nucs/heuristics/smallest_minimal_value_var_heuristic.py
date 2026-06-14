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

from nucs.constants import MAX, MIN


@njit(cache=True, fastmath=True)
def smallest_minimal_value_var_heuristic(
    decision_variables: NDArray, domains_stk: NDArray, top: int, params: NDArray
) -> int:
    """
    Chooses the first variable which is not instantiated with the smallest minimal value.

    :param decision_variables: the decision variables
    :type decision_variables: NDArray
    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the variable
    :rtype: int
    """
    best_min = sys.maxsize
    best_variable = -1
    for variable in decision_variables:
        domain = domains_stk[top, variable]
        if domain[MIN] < domain[MAX] and domain[MIN] < best_min:  # not instantiated
            best_variable = variable
            best_min = domain[MIN]
    return best_variable
