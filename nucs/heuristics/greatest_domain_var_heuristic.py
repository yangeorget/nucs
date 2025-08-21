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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def greatest_domain_var_heuristic(
    decision_variables: NDArray, domains_stk: NDArray, stks_top: NDArray, params: NDArray
) -> int:
    """
    Chooses the first variable which is not instantiated with the greatest domain.
    :param decision_variables: the decision variables
    :param domains_stk: the stack of domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param params: a two-dimensional parameters array, unused here
    :return: the variable
    """
    best_score = 0
    best_variable = -1
    top = stks_top[0]
    for variable in decision_variables:
        domain = domains_stk[top, variable]
        score = domain[MAX] - domain[MIN]  # this is size - 1
        if best_score < score:
            best_variable = variable
            best_score = score
    return best_variable
