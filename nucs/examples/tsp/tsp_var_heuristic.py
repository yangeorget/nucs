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

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN
from nucs.heuristics.max_regret_var_heuristic import regret


@njit(cache=True)
def tsp_var_heuristic(
    decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray, params: NDArray
) -> int:
    """
    :param decision_domains: the indices of a subset of the shared domains
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param params: a two-dimensional (first dimension correspond to variables, second to values) costs array
    :return: the index of the shared domain
    """
    best_score = -sys.maxsize
    best_idx = -1
    top = stacks_top[0]
    for dom_idx in decision_domains:
        shr_domain = shr_domains_stack[top, dom_idx]
        if 0 < shr_domain[MAX] - shr_domain[MIN]:
            score = compute_score(shr_domain, dom_idx, params)
            if best_score < score:
                best_idx = dom_idx
                best_score = score
    return best_idx


@njit(cache=True)
def compute_score(shr_domain: NDArray, dom_idx: int, params: NDArray) -> int:
    """
    Minimize [min(5, size(X)), -regret(X)] for lexicographic order.
    """
    size = min(12, shr_domain[MAX] - shr_domain[MIN] + 1)
    return -size * 1024 + regret(shr_domain, params[dom_idx])
