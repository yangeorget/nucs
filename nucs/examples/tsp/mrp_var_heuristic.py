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
import sys

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True)
def mrp_var_heuristic(
    decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray, params: NDArray
) -> int:
    """
    Chooses the variable with the maximal regret (difference between best and second-best value).
    :param decision_domains: the indices of a subset of the shared domains
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param params: a two-dimensional (first dimension correspond to variables, second to values) costs array
    :return: the index of the shared domain
    """
    top = stacks_top[0]
    n = len(decision_domains)
    used = np.zeros(n, dtype=np.bool)
    for i in range(n):
        if shr_domains_stack[top, i, MIN] == shr_domains_stack[top, i, MAX]:
            used[shr_domains_stack[top, i, MIN]] = True
    max_regret = 0
    best_idx = -1
    for dom_idx in decision_domains:
        shr_domain = shr_domains_stack[top, dom_idx]
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if 0 < size:
            best_cost = second_cost = sys.maxsize
            for value in range(shr_domain[MIN], shr_domain[MAX] + 1):
                if not used[value]:
                    cost = params[dom_idx][value]
                    if cost > 0:
                        if cost < best_cost:
                            second_cost = best_cost
                            best_cost = cost
                        elif cost < second_cost:
                            second_cost = cost
            regret = second_cost - best_cost
            if max_regret < regret:
                best_idx = dom_idx
                max_regret = regret
    return best_idx
