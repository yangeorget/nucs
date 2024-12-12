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
from nucs.heuristics.value_dom_heuristic import value_dom_heuristic


@njit(cache=True)
def min_cost_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
    params: NDArray,
) -> int:
    """
    Chooses the value that minimizes the cost.
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :param params: a two-dimensional (first dimension correspond to variables, second to values) costs array
    :return: the events
    """
    top = stacks_top[0]
    best_cost = sys.maxsize
    best_value = -1
    shr_domain = shr_domains_stack[top, dom_idx]
    for value in range(shr_domain[MIN], shr_domain[MAX] + 1):
        cost = params[dom_idx][value]
        if 0 < cost < best_cost:
            best_cost = cost
            best_value = value
    return value_dom_heuristic(
        shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx, best_value, params
    )
