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
"""
The propagator weights: how often each propagator failed, which the dom/wdeg variable heuristic reads.

One float64 array holds a weight per propagator, then the increment that the next failure adds, then the factor by
which each failure multiplies that increment:

    [ weight of propagator 0 | ... | weight of propagator P - 1 | increment | growth ]

The weights are global and monotone. Nothing trails them, and neither a backtrack nor a restart resets them: what the
search learns about the constraints stays valid in every part of the tree (Choco and Gecode do the same).

A growth above 1 gives the decay of Gecode's AFC without a pass over all the weights at each failure: to make the old
failures count less is the same as to make the new ones count more. When the increment gets too large, all the
cells are divided by the same factor, which keeps every ratio.
"""

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

WEIGHTS_RESCALE_LIMIT = 1e100  # the increment above which the weights and the increment are rescaled
WEIGHTS_RESCALE_FACTOR = 1e-100


def weights_init(propagator_nb: int, decay: float = 1.0) -> NDArray:
    """
    Allocates the propagator weights: each propagator starts at 1, as each failure adds 1 at first.

    :param propagator_nb: the number of propagators
    :type propagator_nb: int
    :param decay: in (0, 1], how much the weight of a failure keeps after each later failure;
                  1 counts all the failures the same (dom/wdeg), less than 1 prefers the recent ones (AFC)
    :type decay: float

    :return: the weights, followed by the increment and its growth
    :rtype: NDArray
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"The weight decay must be in (0, 1], not {decay}")
    weights = np.ones(propagator_nb + 2, dtype=np.float64)
    weights[propagator_nb + 1] = 1.0 / decay
    return weights


# always inlined: bc_algorithm is compiled without the reference-counting runtime and calls it (see bc_algorithm)
@njit(cache=True, inline="always")
def weights_bump(propagator_weights: NDArray, propagator: int) -> None:
    """
    Records a failure of a propagator.

    :param propagator_weights: the weights, followed by the increment and its growth
    :type propagator_weights: NDArray
    :param propagator: the propagator that failed
    :type propagator: int
    """
    increment_idx = len(propagator_weights) - 2
    increment = propagator_weights[increment_idx]
    propagator_weights[propagator] += increment
    increment *= propagator_weights[increment_idx + 1]
    if increment > WEIGHTS_RESCALE_LIMIT:
        for idx in range(increment_idx):
            propagator_weights[idx] *= WEIGHTS_RESCALE_FACTOR
        increment *= WEIGHTS_RESCALE_FACTOR
    propagator_weights[increment_idx] = increment
