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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.problems.problem import OFFSETS_VARIABLE

LIVENESS_UNKNOWN = 0
LIVENESS_LIVE = 1
LIVENESS_DEAD = 2


@njit(cache=True)
def dom_wdeg_var_heuristic(
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
    Chooses the unbound variable with the smallest ratio of its domain size to its weighted degree
    (dom/wdeg, Boussemart et al. 2004, as in Choco; with a weight decay, Gecode's AFC).

    The weighted degree of a variable is the sum of the weights of its live propagators. A propagator is live when it
    is not entailed and has an unbound variable other than this one: a propagator that can no longer fail on a
    decision about this variable says nothing about it. A variable with no live propagator is chosen only when no
    other variable is left, by smallest domain. Ties go to the first variable in decision order.

    The cost is the number of propagators of the unbound decision variables, plus one scan of each of these
    propagators, since its liveness is computed once.

    :param decision_variables: the decision variables
    :type decision_variables: NDArray
    :param domains: the domains
    :type domains: NDArray
    :param entailed: whether each propagator is entailed
    :type entailed: NDArray
    :param offsets: the offsets of the propagators, whose OFFSETS_VARIABLE column delimits their variables
    :type offsets: NDArray
    :param propagator_variables: the variables of the propagators
    :type propagator_variables: NDArray
    :param variable_propagators_offsets: the offsets of the propagators of each variable
    :type variable_propagators_offsets: NDArray
    :param variable_propagators: the propagators of the variables, each one once per variable
    :type variable_propagators: NDArray
    :param propagator_weights: the failure weights of the propagators
    :type propagator_weights: NDArray
    :param params: a two-dimensional parameter array, unused here
    :type params: NDArray

    :return: the variable, or -1 when all the decision variables are bound
    :rtype: int
    """
    liveness = np.zeros(len(entailed), dtype=np.uint8)
    best_variable = -1
    best_isolated = True  # whether the best variable so far has no live propagator
    best_score = np.inf
    for variable in decision_variables:
        size = np.int64(domains[variable, DOMAIN_MAX]) - np.int64(domains[variable, DOMAIN_MIN])
        if size == 0:
            continue
        wdeg = 0.0
        for idx in range(variable_propagators_offsets[variable], variable_propagators_offsets[variable + 1]):
            propagator = variable_propagators[idx]
            if liveness[propagator] == LIVENESS_UNKNOWN:
                liveness[propagator] = (
                    LIVENESS_LIVE
                    if not entailed[propagator]
                    and has_two_unbound_variables(domains, offsets, propagator_variables, propagator)
                    else LIVENESS_DEAD
                )
            if liveness[propagator] == LIVENESS_LIVE:
                wdeg += propagator_weights[propagator]
        isolated = wdeg == 0.0
        score = float(size + 1) if isolated else (size + 1) / wdeg
        if (best_isolated and not isolated) or (best_isolated == isolated and score < best_score):
            best_variable = variable
            best_isolated = isolated
            best_score = score
    return best_variable


@njit(cache=True)
def has_two_unbound_variables(
    domains: NDArray, offsets: NDArray, propagator_variables: NDArray, propagator: int
) -> bool:
    """
    Tells whether a propagator has at least two distinct unbound variables.

    :param domains: the domains
    :type domains: NDArray
    :param offsets: the offsets of the propagators
    :type offsets: NDArray
    :param propagator_variables: the variables of the propagators
    :type propagator_variables: NDArray
    :param propagator: the propagator
    :type propagator: int

    :return: True when the propagator has two distinct unbound variables
    :rtype: bool
    """
    first_unbound = -1
    for idx in range(offsets[propagator, OFFSETS_VARIABLE], offsets[propagator + 1, OFFSETS_VARIABLE]):
        variable = propagator_variables[idx]
        if domains[variable, DOMAIN_MIN] < domains[variable, DOMAIN_MAX]:
            if first_unbound == -1:
                first_unbound = variable
            elif variable != first_unbound:
                return True
    return False
