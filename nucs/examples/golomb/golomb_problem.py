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
import math
from typing import Any

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_GROUND, EVENT_MASK_MIN, EVENT_MASK_NB, MAX, MIN
from nucs.numba_helper import NDArrayList, ComputeDomainsFunctions
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    ALG_ALLDIFFERENT,
    ALG_LEQ_C,
    ALG_SUM_EQ,
    update_propagators,
)
from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm, first_unbound_decision_variable

GOLOMB_LENGTHS = np.array([0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127])


@njit(cache=True, fastmath=True)
def sum_first(n: int) -> int:
    """
    Returns the sum of the first n integers.

    :param n: an integer
    :type n: int

    :return: the sum
    :rtype: int
    """
    return (n * (n + 1)) >> 1


@njit(cache=True, fastmath=True)
def index(mark_nb: int, i: int, j: int) -> int:
    """
    Returns the index of the distance variable between two marks.

    :param mark_nb: the total number of marks
    :type mark_nb: int
    :param i: the first mark
    :type i: int
    :param j: the second mark
    :type j: int

    :return: the index of the distance variable
    :rtype: int
    """
    return i * mark_nb - sum_first(i) + j - i - 1


class GolombProblem(Problem):
    """
    This is the famous Golomb ruler problem.

    It consists in finding n integers mark_i such that:
    - mark_0 = 0,
    - mark_0 <...< mark_n-1,
    - for all i < j, mark_j - mark_i are different,
    - mark_n-1 is minimal.

    CSPLIB problem #6 - https://www.csplib.org/Problems/prob006/
    """

    def __init__(self, mark_nb: int, symmetry_breaking: bool = True) -> None:
        """
        Initializes the problem.

        :param mark_nb: the number of marks
        :type mark_nb: int
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        :type symmetry_breaking: bool
        """
        self.mark_nb = mark_nb
        dist_nb = sum_first(mark_nb - 1)
        domains = [[0, sum_first(dist_nb) - sum_first(dist_nb - mark_nb)]] * dist_nb
        for i in range(0, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                domains[index(mark_nb, i, j)][MIN] = (
                    GOLOMB_LENGTHS[j - i + 1] if j - i + 1 < mark_nb else sum_first(j - i)
                )
        super().__init__([(domain[MIN], domain[MAX]) for domain in domains])
        self.length_idx = index(mark_nb, 0, mark_nb - 1)  # we want to minimize this
        # dist_ij = mark_j - mark_i for j > i
        # mark_j = dist_0j for j > 0
        for i in range(1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                self.add_propagator(ALG_SUM_EQ, [index(mark_nb, 0, i), index(mark_nb, i, j), index(mark_nb, 0, j)])
        self.add_propagator(ALG_ALLDIFFERENT, range(self.domain_nb))
        # redundant constraints
        for i in range(mark_nb - 1):
            for j in range(i + 1, mark_nb):
                if j - i < mark_nb - 1:
                    self.add_propagator(
                        ALG_LEQ_C,
                        [index(mark_nb, i, j), index(mark_nb, 0, mark_nb - 1)],
                        [-sum_first(mark_nb - 1 - (j - i))],
                    )
        if symmetry_breaking:
            self.add_propagator(
                ALG_LEQ_C,
                [index(mark_nb, 0, 1), index(mark_nb, mark_nb - 2, mark_nb - 1)],
                [-1],
            )

    def solution_as_printable(self, solution: NDArray) -> Any:
        solution_as_list = solution.tolist()
        return solution_as_list[: self.mark_nb]


@njit(cache=True, fastmath=True)
def golomb_consistency_algorithm(
    algorithm_nb: int,
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    complexities: NDArray,
    bounds: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_fcts: ComputeDomainsFunctions,
    decision_variables: NDArrayList,
    domain_buffer: NDArray,
) -> int:
    """
    Applies a custom consistency algorithm for the Golomb Ruler problem.

    :param statistics: the statistics array
    :type statistics: NDArray

    :return: the status as an int
    :rtype: int
    """
    top = stks_top[0]
    # first prune the search space
    domain_nb = (len(triggers_offsets) - 1) // EVENT_MASK_NB
    mark_nb = (1 + int(math.sqrt(8 * domain_nb + 1))) >> 1
    ni_var = first_unbound_decision_variable(decision_variables, domains_stk, top, 0)
    if 1 < ni_var < mark_nb - 1:  # otherwise useless
        used_distance = np.zeros(sum_first(mark_nb - 2) + 1, dtype=np.bool)
        # a reusable array for storing the minimal sum of different integers:
        # minimal_sum[i] will be the minimal sum of i different integers chosen among a set of possible integers
        minimal_sum = np.zeros(mark_nb - 1, dtype=np.int32)
        # the following will mark at most sum(n-3) numbers as used
        # hence there will be at least n-2 unused numbers greater than 0
        for var in range(index(mark_nb, ni_var - 2, ni_var - 1) + 1):
            dist = domains_stk[top, var, MIN]  # no offset
            if dist < len(used_distance):
                used_distance[dist] = True
        # let's compute the sum of non-used numbers
        distance = 1
        for j in range(0, mark_nb - ni_var):
            while used_distance[distance]:
                distance += 1
            minimal_sum[j + 1] = minimal_sum[j] + distance
            distance += 1
        for i in range(ni_var - 1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                var = index(mark_nb, i, j)
                old_min = domains_stk[top, var, MIN]
                domains_stk[top, var, MIN] = max(old_min, minimal_sum[j - i])  # no offset
                if domains_stk[top, var, MIN] != old_min:
                    events = EVENT_MASK_MIN
                    if domains_stk[top, var, MIN] == domains_stk[top, var, MAX]:
                        events |= EVENT_MASK_GROUND
                        unbound_variable_nb_stk[top] -= 1
                    offset = var * EVENT_MASK_NB + events
                    update_propagators(
                        triggered_propagators,
                        entailed_propagator_depths,
                        triggers[triggers_offsets[offset] : triggers_offsets[offset + 1]],
                        complexities,
                        propagator_nb,
                    )
    return bound_consistency_algorithm(
        algorithm_nb,
        propagator_nb,
        statistics,
        algorithms,
        complexities,
        bounds,
        propagator_variables,
        propagator_parameters,
        triggers,
        triggers_offsets,
        domains_stk,
        entailed_propagator_depths,
        entailment_trail,
        domain_update_stk,
        unbound_variable_nb_stk,
        stks_top,
        triggered_propagators,
        compute_domains_fcts,
        decision_variables,
        domain_buffer,
    )
