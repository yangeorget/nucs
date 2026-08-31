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

from nucs.constants import DECISION_EQ, MAX, MIN


@njit(cache=True)
def random_value_dom_heuristic(domains: NDArray, variable: int, params: NDArray) -> tuple[int, int]:
    """
    Chooses a value uniformly at random in the domain.

    This is FlatZinc's indomain_random. Domains are intervals here, so every value between the two bounds
    is in the domain and drawing uniformly between them needs no rejection.

    The draw is a decision, not a restriction: DECISION_EQ parks the values below and above the one it
    picks, so the enumeration stays exhaustive and only the order in which it is explored is random.

    :param domains: the domains
    :type domains: NDArray
    :param variable: the variable
    :type variable: int
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the kind of the decision and the value the domain is split at
    :rtype: Tuple[int, int]
    """
    return DECISION_EQ, random.randint(domains[variable, MIN], domains[variable, MAX])
