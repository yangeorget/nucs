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
def split_random_dom_heuristic(domains: NDArray, variable: int, params: NDArray) -> tuple[int, int]:
    """
    Chooses at random the first or the second half of the domain.

    :param domains: the domains
    :type domains: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the kind of the decision and the value the domain is split at
    :rtype: Tuple[int, int]
    """
    if random.randint(0, 1) == 0:
        return split_low_dom_heuristic(domains, variable, params)
    return split_high_dom_heuristic(domains, variable, params)
