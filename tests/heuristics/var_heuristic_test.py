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
from collections.abc import Callable, Sequence

import numpy as np

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_DUMMY


def call_var_heuristic(
    var_heuristic: Callable,
    domains: Sequence[tuple[int, int]],
    decision_variables: Sequence[int] | None = None,
    propagators: Sequence[Sequence[int]] = (),
    entailed: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
) -> int:
    """
    Calls a variable heuristic on a network built from its description, as the solver would.

    :param var_heuristic: the variable heuristic
    :type var_heuristic: Callable
    :param domains: the domain of each variable
    :type domains: Sequence[Tuple[int, int]]
    :param decision_variables: the decision variables, defaults to all the variables
    :type decision_variables: Optional[Sequence[int]]
    :param propagators: the variables of each propagator, whose filtering is irrelevant here
    :type propagators: Sequence[Sequence[int]]
    :param entailed: whether each propagator is entailed, defaults to none
    :type entailed: Optional[Sequence[int]]
    :param weights: the failure weight of each propagator, defaults to 1
    :type weights: Optional[Sequence[float]]

    :return: the variable the heuristic chooses
    :rtype: int
    """
    problem = Problem(domains)
    for variables in propagators:
        problem.add_propagator(ALG_DUMMY, variables)
    problem.init()
    propagator_nb = problem.propagator_nb
    if decision_variables is None:
        decision_variables = range(problem.domain_nb)
    propagator_weights = np.ones(propagator_nb + 2, dtype=np.float64)  # the weights, the increment and its growth
    if weights is not None:
        propagator_weights[:propagator_nb] = weights
    return int(
        var_heuristic(
            np.array(decision_variables, dtype=np.uint32),
            problem.initial_domains,
            np.array(entailed if entailed is not None else [0] * propagator_nb, dtype=np.int32),
            problem.offsets,
            problem.propagator_variables,
            problem.variable_propagators_offsets,
            problem.variable_propagators,
            propagator_weights,
            np.empty((1, 0), dtype=np.int64),
        )
    )
