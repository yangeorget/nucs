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
# Copyright 2024 - Yan Georget
###############################################################################
import math

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_GROUND, EVENT_MASK_MIN, MAX, MIN
from nucs.heuristics.heuristics import first_not_instantiated_var_heuristic
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_AFFINE_LEQ, ALG_ALLDIFFERENT, update_propagators
from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm

GOLOMB_LENGTHS = np.array([0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127])


@njit(cache=True)
def sum_first(n: int) -> int:
    """
    Returns the sum of the first n integers.
    :param n: an integer
    :return: the sum
    """
    return (n * (n + 1)) // 2


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
        dist_nb = sum_first(mark_nb - 1)
        domains = np.empty((dist_nb, 2), dtype=np.int32)
        for i in range(0, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                domains[index(mark_nb, i, j), MIN] = (
                    GOLOMB_LENGTHS[j - i + 1] if j - i + 1 < mark_nb else sum_first(j - i)
                )
        domains[:, MAX] = sum_first(dist_nb) - sum_first(dist_nb - mark_nb)
        super().__init__(domains)
        self.length_idx = index(mark_nb, 0, mark_nb - 1)  # we want to minimize this
        # dist_ij = mark_j - mark_i for j > i
        # mark_j = dist_0j for j > 0
        for i in range(1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                self.add_propagator(
                    (
                        [index(mark_nb, 0, j), index(mark_nb, 0, i), index(mark_nb, i, j)],
                        ALG_AFFINE_EQ,
                        [1, -1, -1, 0],
                    )
                )
        self.add_propagator((list(range(self.shr_domain_nb)), ALG_ALLDIFFERENT, []))
        # redundant constraints
        for i in range(mark_nb - 1):
            for j in range(i + 1, mark_nb):
                if j - i < mark_nb - 1:
                    self.add_propagator(
                        (
                            [index(mark_nb, i, j), index(mark_nb, 0, mark_nb - 1)],
                            ALG_AFFINE_LEQ,
                            [1, -1, -sum_first(mark_nb - 1 - (j - i))],
                        )
                    )
        if symmetry_breaking:
            self.add_propagator(
                (
                    [index(mark_nb, 0, 1), index(mark_nb, mark_nb - 2, mark_nb - 1)],
                    ALG_AFFINE_LEQ,
                    [1, -1, -1],
                )
            )


@njit(cache=True)
def golomb_consistency_algorithm(
    statistics: NDArray,
    algorithms: NDArray,
    var_bounds: NDArray,
    param_bounds: NDArray,
    dom_indices_arr: NDArray,
    dom_offsets_arr: NDArray,
    props_dom_indices: NDArray,
    props_dom_offsets: NDArray,
    props_parameters: NDArray,
    shr_domains_propagators: NDArray,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    decision_domains: NDArray,
) -> int:
    """
    Applies a custom consistency algorithm for the Golomb Ruler problem.
    :param statistics: the statistics array
    :param problem: the problem
    :return: the status as an int
    """
    top = stacks_top[0]
    # first prune the search space
    mark_nb = (1 + int(math.sqrt(8 * len(dom_indices_arr) + 1))) // 2
    ni_var_idx = first_not_instantiated_var_heuristic(
        decision_domains, shr_domains_stack, stacks_top, None
    )  # no domains shared between vars
    if 1 < ni_var_idx < mark_nb - 1:  # otherwise useless
        used_distance = np.zeros(sum_first(mark_nb - 2) + 1, dtype=np.bool)
        # a reusable array for storing the minimal sum of different integers:
        # minimal_sum[i] will be the minimal sum of i different integers chosen among a set of possible integers
        minimal_sum = np.zeros(mark_nb - 1, dtype=np.int32)
        # the following will mark at most sum(n-3) numbers as used
        # hence there will be at least n-2 unused numbers greater than 0
        for var_idx in range(index(mark_nb, ni_var_idx - 2, ni_var_idx - 1) + 1):
            dist = shr_domains_stack[top, dom_indices_arr[var_idx], MIN]  # no offset
            if dist < len(used_distance):
                used_distance[dist] = True
        # let's compute the sum of non-used numbers
        distance = 1
        for j in range(0, mark_nb - ni_var_idx):
            while used_distance[distance]:
                distance += 1
            minimal_sum[j + 1] = minimal_sum[j] + distance
            distance += 1
        for i in range(ni_var_idx - 1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                dom_idx = dom_indices_arr[index(mark_nb, i, j)]
                shr_domains_stack[top, dom_idx, MIN] = minimal_sum[j - i]  # no offset
                events = EVENT_MASK_MIN
                if shr_domains_stack[top, dom_idx, MIN] == shr_domains_stack[top, dom_idx, MAX]:
                    events |= EVENT_MASK_GROUND
                update_propagators(
                    triggered_propagators,
                    not_entailed_propagators_stack[top],
                    shr_domains_propagators,
                    dom_idx,
                    events,
                )
    return bound_consistency_algorithm(
        statistics,
        algorithms,
        var_bounds,
        param_bounds,
        dom_indices_arr,
        dom_offsets_arr,
        props_dom_indices,
        props_dom_offsets,
        props_parameters,
        shr_domains_propagators,
        shr_domains_stack,
        not_entailed_propagators_stack,
        dom_update_stack,
        stacks_top,
        triggered_propagators,
        compute_domains_addrs,
        decision_domains,
    )
