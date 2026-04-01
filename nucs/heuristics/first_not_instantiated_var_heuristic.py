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

from nucs.constants import MAX, MIN


@njit(cache=True)
def first_not_instantiated_var_heuristic(
    decision_variables: NDArray, domains_stk: NDArray, top: int, params: NDArray
) -> int:
    """
    Chooses the first non-instantiated variable.
    :param decision_variables: the decision variables
    :param domains_stk: the stack of domains
    :param top: the index of the top of the stacks
    :param params: a two-dimensional parameter array, unused here
    :return: the variable
    """
    for variable in decision_variables:
        if domains_stk[top, variable, MIN] < domains_stk[top, variable, MAX]:
            return variable
    return -1  # cannot happen
