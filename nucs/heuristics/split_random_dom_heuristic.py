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


@njit(cache=True, fastmath=True)
def split_random_dom_heuristic(
    domains_stk: NDArray,
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    params: NDArray,
) -> int:
    return (
        split_low_dom_heuristic(
            domains_stk,
            entailed_propagators_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            params,
        )
        if random.randint(0, 1) == 0
        else split_high_dom_heuristic(
            domains_stk,
            entailed_propagators_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            variable,
            params,
        )
    )
