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
import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN


@njit(cache=True)
def largest_maximal_value_var_heuristic(
    decision_variables: NDArray,
    domains: NDArray,
    entailed: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    variable_propagators_offsets: NDArray,
    variable_propagators: NDArray,
    propagator_weights: NDArray,
    params: NDArray,
) -> int:
    """
    Chooses the first variable which is not instantiated with the largest maximal value.

    :param decision_variables: the decision variables
    :type decision_variables: NDArray
    :param domains: the domains
    :type domains: NDArray
    :param entailed: whether each propagator is entailed, unused here
    :type entailed: NDArray
    :param offsets: the offsets of the propagators, unused here
    :type offsets: NDArray
    :param propagator_variables: the variables of the propagators, unused here
    :type propagator_variables: NDArray
    :param variable_propagators_offsets: the offsets of the propagators of each variable, unused here
    :type variable_propagators_offsets: NDArray
    :param variable_propagators: the propagators of the variables, unused here
    :type variable_propagators: NDArray
    :param propagator_weights: the failure weights of the propagators, unused here
    :type propagator_weights: NDArray
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the variable
    :rtype: int
    """
    best_max = -sys.maxsize
    best_variable = -1
    for variable in decision_variables:
        domain = domains[variable]
        if domain[DOMAIN_MIN] < domain[DOMAIN_MAX] and domain[DOMAIN_MAX] > best_max:  # not instantiated
            best_variable = variable
            best_max = domain[DOMAIN_MAX]
    return best_variable
