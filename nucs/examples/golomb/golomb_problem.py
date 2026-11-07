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

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_NB
from nucs.problems.problem import PROBLEM_INCONSISTENT, Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_LEQ_C, ALG_SUM_EQ
from nucs.solvers.backtrack.bc_algorithm import bc_algorithm
from nucs.solvers.choicepoints import CHOICEPOINT_UTRAIL_MARK
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.state import tighten_at

GOLOMB_LENGTHS = np.array([0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127])


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
        for i in range(mark_nb - 1):
            for j in range(i + 1, mark_nb):
                domains[index(mark_nb, i, j)][DOMAIN_MIN] = (
                    GOLOMB_LENGTHS[j - i + 1] if j - i + 1 < mark_nb else sum_first(j - i)
                )
        super().__init__([(domain[DOMAIN_MIN], domain[DOMAIN_MAX]) for domain in domains])
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


@njit(cache=True)
def sum_first(n: int) -> int:
    """
    Returns the sum of the first n integers.

    :param n: an integer
    :type n: int

    :return: the sum
    :rtype: int
    """
    return (n * (n + 1)) >> 1


@njit(cache=True)
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


@njit(cache=True)
def golomb_consistency_algorithm(
    statistics: NDArray,
    propagator_weights: NDArray,
    algorithm_flags: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    state: NDArray,
    domains: NDArray,
    entailed: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    domain_buffer: NDArray,
) -> int:
    """
    Applies a custom consistency algorithm for the Golomb Ruler problem.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param propagator_weights: the propagator weights, forwarded to bc_algorithm
    :type propagator_weights: NDArray
    :param algorithm_flags: the PROP_FLAG_* properties of each algorithm, packed into one word and indexed by
                          algorithm rather than by propagator
    :type algorithm_flags: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param offsets: the CSR offsets delimiting each propagator's slice of propagator_variables,
                    propagator_parameters and its own state block, whose trailed prefix ends at
                    OFFSETS_STATE_HINT (see OFFSETS_STATE)
    :type offsets: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param state: all the backtrackable state: the domain bounds, the unbound-variable count and,
                  from there on, whatever else is trailed
    :type state: NDArray
    :param domains: the current domains, a (domain_nb, 2) view of the head of state
    :type domains: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_addrs: the compiled compute_domains address of each algorithm, resolved once at solver init
    :type compute_domains_addrs: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray

    :return: the status as an int
    :rtype: int
    """
    mark = choicepoints[choicepoint_top[0], CHOICEPOINT_UTRAIL_MARK]
    # as in the propagation loop, the undo trail size stays in a register across the whole pruning
    utrail_size = utrail_top[0]
    # first prune the search space
    domain_nb = (len(triggers_offsets) - 1) >> EVENT_NB
    mark_nb = (1 + int(math.sqrt(8 * domain_nb + 1))) >> 1
    # the first unbound variable in index order (the marks are the leading variables), -1 when all are bound
    ni_var = -1
    for var in range(domain_nb):
        if domains[var, DOMAIN_MIN] < domains[var, DOMAIN_MAX]:
            ni_var = var
            break
    if 1 < ni_var < mark_nb - 1:  # otherwise useless
        used_distance = np.zeros(sum_first(mark_nb - 2) + 1, dtype=np.bool)
        # a reusable array for storing the minimal sum of different integers:
        # minimal_sum[i] will be the minimal sum of i different integers chosen among a set of possible integers
        minimal_sum = np.zeros(mark_nb - 1, dtype=np.int32)
        # the distances between the marks 0..ni_var-1, which are bound (mark k is variable k-1): the following
        # will mark at most sum(n-3) numbers as used, hence there will be at least n-2 unused numbers greater than 0;
        # a distance is read from its two marks, because bc_algorithm has not yet propagated them into the distance
        # variables, and a variable of a later mark can still be unbound
        for i in range(ni_var - 1):
            mark_i = 0 if i == 0 else domains[i - 1, DOMAIN_MIN]  # no offset
            for j in range(i + 1, ni_var):
                dist = domains[j - 1, DOMAIN_MIN] - mark_i
                if dist < len(used_distance):
                    used_distance[dist] = True
        # let's compute the sum of non-used numbers
        distance = 1
        for j in range(mark_nb - ni_var):
            while used_distance[distance]:
                distance += 1
            minimal_sum[j + 1] = minimal_sum[j] + distance
            distance += 1
        for i in range(ni_var - 1, mark_nb - 1):
            for j in range(i + 1, mark_nb):
                var = index(mark_nb, i, j)
                cell_idx = var << 1
                new_min = max(state[cell_idx], minimal_sum[j - i])  # no offset
                # tighten leaves the wipeout test to its caller: an empty domain must fail here, because the
                # propagators of bc_algorithm do not expect one (a search that is not in index order meets it)
                if new_min > state[cell_idx | 1]:
                    utrail_top[0] = utrail_size
                    return PROBLEM_INCONSISTENT
                # every domain write goes through tighten, which owns the groundness test and the
                # unbound-variable count -- and, once domains are trailed, the write barrier
                events, utrail_size = tighten_at(
                    state,
                    utrail_log,
                    utrail_indices,
                    mark,
                    utrail_size,
                    var,
                    new_min,
                    state[cell_idx | 1],
                )
                if events:
                    update_propagators(
                        triggered_propagators,
                        entailed,
                        triggers,
                        triggers_offsets,
                        priorities,
                        var,
                        events,
                        -1,
                    )
    utrail_top[0] = utrail_size
    return bc_algorithm(
        statistics,
        propagator_weights,
        algorithm_flags,
        algorithms,
        priorities,
        offsets,
        propagator_variables,
        propagator_parameters,
        triggers,
        triggers_offsets,
        state,
        domains,
        entailed,
        utrail_log,
        utrail_top,
        utrail_indices,
        choicepoints,
        choicepoint_top,
        triggered_propagators,
        compute_domains_addrs,
        domain_buffer,
    )
