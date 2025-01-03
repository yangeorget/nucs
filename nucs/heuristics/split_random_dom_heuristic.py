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
import random

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.heuristics.split_high_dom_heuristic import split_high_dom_heuristic
from nucs.heuristics.split_low_dom_heuristic import split_low_dom_heuristic


@njit(cache=True)
def split_random_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
    params: NDArray,
) -> int:
    if random.randint(0, 1) == 0:
        return split_low_dom_heuristic(
            shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx, params
        )
    else:
        return split_high_dom_heuristic(
            shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx, params
        )
