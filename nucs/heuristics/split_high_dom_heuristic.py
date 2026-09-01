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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DECISION_GT, DOMAIN_MAX, DOMAIN_MIN


@njit(cache=True)
def split_high_dom_heuristic(domains: NDArray, variable: int, params: NDArray) -> tuple[int, int]:
    """
    Chooses the second half of the domain.

    :param domains: the domains
    :type domains: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the kind of the decision and the value the domain is split at
    :rtype: Tuple[int, int]
    """
    return DECISION_GT, (domains[variable, DOMAIN_MIN] + domains[variable, DOMAIN_MAX]) >> 1
