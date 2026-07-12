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
import random

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.heuristics.split_high_dom_heuristic import split_high_dom_heuristic
from nucs.heuristics.split_low_dom_heuristic import split_low_dom_heuristic


@njit(cache=True)
def split_random_dom_heuristic(
    domains_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    params: NDArray,
) -> int:
    """
    Chooses at random the first or the second half of the domain.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the events
    :rtype: int
    """
    return (
        split_low_dom_heuristic(
            domains_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            params,
        )
        if random.randint(0, 1) == 0
        else split_high_dom_heuristic(
            domains_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            params,
        )
    )
