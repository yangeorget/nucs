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
# Copyright 2024-2025 - Yan Georget
###############################################################################
import math
from typing import Any

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_GROUND, EVENT_MASK_MIN, MAX, MIN
from nucs.heuristics.heuristics import first_not_instantiated_var_heuristic
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_LEQ, ALG_SUM_EQ, update_propagators
from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm

GOLOMB_LENGTHS = np.array([0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127])


@njit(cache=True)
def sum_first(n: int) -> int:
    """
    Returns the sum of the first n integers.
    :param n: an integer
    :return: the sum
    """
    return (n * (n + 1)) >> 1


@njit(cache=True)
def index(mark_nb: int, i: int, j: int) -> int:
    """
    Returns the index of the distance variable between two marks.
    :param mark_nb: the total number of marks
    :param i: the first mark
    :param j: the second mark
    :return: the index of the distance variable
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
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
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
                        ALG_LEQ,
                        [index(mark_nb, i, j), index(mark_nb, 0, mark_nb - 1)],
                        [-sum_first(mark_nb - 1 - (j - i))],
                    )
        if symmetry_breaking:
            self.add_propagator(
                ALG_LEQ,
                [index(mark_nb, 0, 1), index(mark_nb, mark_nb - 2, mark_nb - 1)],
                [-1],
            )

    def solution_as_printable(self, solution: NDArray) -> Any:
        solution_as_list = solution.tolist()
        return solution_as_list[: self.mark_nb]


@njit(cache=True)
def golomb_consistency_algorithm(
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    bounds: NDArray,
    props_variables: NDArray,
    props_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    decision_variables: NDArray,
) -> int:
    """
    Applies a custom consistency algorithm for the Golomb Ruler problem.
    :param statistics: the statistics array
    :return: the status as an int
    """
    top = stks_top[0]
    # first prune the search space
    mark_nb = (1 + int(math.sqrt(8 * len(triggers) + 1))) // 2
    ni_var = first_not_instantiated_var_heuristic(
        decision_variables, domains_stk, stks_top, None
    )  # no domains shared between vars
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
                domains_stk[top, var, MIN] = minimal_sum[j - i]  # no offset
                events = EVENT_MASK_MIN
                if domains_stk[top, var, MIN] == domains_stk[top, var, MAX]:
                    events |= EVENT_MASK_GROUND
                update_propagators(
                    propagator_nb, triggered_propagators, not_entailed_propagators_stk[top], triggers, events, var
                )
    return bound_consistency_algorithm(
        propagator_nb,
        statistics,
        algorithms,
        bounds,
        props_variables,
        props_parameters,
        triggers,
        domains_stk,
        not_entailed_propagators_stk,
        dom_update_stk,
        stks_top,
        triggered_propagators,
        compute_domains_addrs,
        decision_variables,
    )
