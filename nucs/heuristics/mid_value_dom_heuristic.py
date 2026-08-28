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

from nucs.constants import DECISION_EQ, DECISION_VALUE, MAX, MIN


@njit(cache=True)
def mid_value_dom_heuristic(domains: NDArray, variable: int, params: NDArray, decision: NDArray) -> int:
    """
    Chooses the middle value of the domain.

    :param domains: the domains
    :type domains: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray
    :param decision: the decision, written by this function
    :type decision: NDArray

    :return: the kind of the decision
    :rtype: int
    """
    decision[DECISION_VALUE] = (domains[variable, MIN] + domains[variable, MAX]) >> 1
    return DECISION_EQ
