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
import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def max_regret_var_heuristic(
    decision_variables: NDArray, domains_stk: NDArray, stks_top: NDArray, params: NDArray
) -> int:
    """
    Chooses the variable with the maximal regret (difference between the best and second-best value).
    :param decision_variables: the decision variables
    :param domains_stk: the stack of domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param params: a two-dimensional (first dimension corresponds to variables, second to values) costs array
    :return: the variable
    """
    best_score = 0
    best_variable = -1
    top = stks_top[0]
    for variable in decision_variables:
        domain = domains_stk[top, variable]
        if 0 < domain[MAX] - domain[MIN]:
            score = regret(domain, params[variable])
            if best_score < score:
                best_variable = variable
                best_score = score
    return best_variable


@njit(cache=True)
def regret(domain: NDArray, costs: NDArray) -> int:
    best_cost = sys.maxsize
    second_cost = sys.maxsize
    for value in range(domain[MIN], domain[MAX] + 1):
        cost = costs[value]
        if 0 < cost < best_cost:
            second_cost = best_cost
            best_cost = cost
        elif 0 < cost < second_cost:
            second_cost = cost
    return second_cost - best_cost
