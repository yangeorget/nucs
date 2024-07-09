from typing import Optional, Tuple, Union

import numpy as np
from numba import jit  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

from ncs.propagators import (
    alldifferent_lopez_ortiz_propagator,
    constant_sum_propagator,
    dummy_propagator,
    mul_propagator,
    sum_propagator,
)
from ncs.utils import (
    MAX,
    MIN,
    STATS_PROBLEM_FILTERS_NB,
    STATS_PROBLEM_PROPAGATORS_FILTERS_NB,
    statistics_init,
)

ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ = 0
ALGORITHM_CONSTANT_SUM = 1
ALGORITHM_DUMMY = 2
ALGORITHM_MUL = 3
ALGORITHM_SUM = 4

START = 0
END = 1


class Problem:
    """
    A problem is conceptually defined by a list of domains, a list of variables and a list of propagators.
    """

    def __init__(
        self, shared_domains: List[Union[int, Tuple[int, int]]], domain_indices: List[int], domain_offsets: List[int]
    ):
        self.variable_nb = len(domain_indices)
        self.shared_domains = np.array(self.build_domains(shared_domains), dtype=np.int32, order="C")
        self.domain_indices = np.array(domain_indices, dtype=np.uint16)
        self.domain_offsets = np.array(domain_offsets, dtype=np.int32)

    def build_domains(self, domains: List[Union[int, Tuple[int, int]]]) -> List[Tuple[int, int]]:
        return [[domain, domain] if isinstance(domain, int) else domain for domain in domains]

    def set_propagators(self, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
        """
        Sets the propagators for the problem.
        :param propagators: the list of propagators as tuples of the form (list of variables, algorithm, data).
        """
        self.propagator_nb = len(propagators)
        propagator_variable_total_size = 0
        propagator_data_total_size = 0
        self.propagators_to_filter = np.empty(self.propagator_nb, dtype=np.bool)
        self.propagators_to_filter = np.empty(self.propagator_nb, dtype=np.bool)
        self.propagator_algorithms = np.empty(self.propagator_nb, dtype=np.uint8)
        self.propagator_variable_bounds = np.empty((self.propagator_nb, 2), dtype=np.uint16, order="C")
        self.propagator_data_bounds = np.empty((self.propagator_nb, 2), dtype=np.uint16, order="C")
        self.propagator_variable_bounds[0, START] = 0
        self.propagator_data_bounds[0, START] = 0
        for prop_idx, propagator in enumerate(propagators):
            propagator_variable_size = len(propagator[0])
            propagator_data_size = len(propagator[2])
            self.propagator_algorithms[prop_idx] = propagator[1]
            if prop_idx > 0:
                self.propagator_variable_bounds[prop_idx, START] = self.propagator_variable_bounds[prop_idx - 1, END]
                self.propagator_data_bounds[prop_idx, START] = self.propagator_data_bounds[prop_idx - 1, END]
            self.propagator_variable_bounds[prop_idx, END] = (
                self.propagator_variable_bounds[prop_idx, START] + propagator_variable_size
            )
            self.propagator_data_bounds[prop_idx, END] = (
                self.propagator_data_bounds[prop_idx, START] + propagator_data_size
            )
            propagator_variable_total_size += propagator_variable_size
            propagator_data_total_size += propagator_data_size
        self.propagator_indices = np.empty(propagator_variable_total_size, dtype=np.uint16)
        self.propagator_offsets = np.empty(propagator_variable_total_size, dtype=np.int32)
        self.propagator_data = np.empty(propagator_data_total_size, dtype=np.int32)
        for prop_idx, propagator in enumerate(propagators):
            prop_variables = propagator[0]
            prop_variable_bounds = self.propagator_variable_bounds[prop_idx]
            self.propagator_indices[prop_variable_bounds[START] : prop_variable_bounds[END]] = self.domain_indices[
                prop_variables
            ]
            self.propagator_offsets[prop_variable_bounds[START] : prop_variable_bounds[END]] = self.domain_offsets[
                prop_variables
            ]
            prop_data_bounds = self.propagator_data_bounds[prop_idx]
            self.propagator_data[prop_data_bounds[START] : prop_data_bounds[END]] = propagator[2]

    def get_values(self) -> List[int]:
        """
        Gets the values for the variables (when instantiated).
        :return: a list of integers
        """
        mins = self.shared_domains[self.domain_indices, MIN]
        mins += self.domain_offsets
        return mins.tolist()

    def __str__(self) -> str:
        return f"domains={self.shared_domains}"

    def filter(self, statistics: NDArray = statistics_init(), changes: Optional[NDArray] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param statistics: where to record the statistics of the computation
        :param changes: an optional array of shared domain changes
        :return: False if the problem is not consistent
        """
        return filter(
            self.propagator_nb,
            self.propagators_to_filter,
            self.propagator_algorithms,
            self.propagator_variable_bounds,
            self.propagator_data_bounds,
            self.propagator_indices,
            self.propagator_offsets,
            self.propagator_data,
            self.shared_domains,
            statistics,
            changes,
        )

    def pretty_print(self, solution: List[int]) -> None:
        """
        Pretty prints a solution to the problem.
        :param solution: a list of integers
        """
        print(solution)


@jit(nopython=True, cache=True)
def filter(
    propagator_nb: int,
    propagators_to_filter: NDArray,
    propagator_algorithms: NDArray,
    propagator_variable_bounds: NDArray,
    propagator_data_bounds: NDArray,
    propagator_indices: NDArray,
    propagator_offsets: NDArray,
    propagator_data: NDArray,
    shared_domains: NDArray,
    statistics: NDArray,
    changes: Optional[NDArray],
) -> bool:
    """
    Filters the problem's domains by applying the propagators until a fix point is reached.
    :param statistics: where to record the statistics of the computation
    :param changes: an optional array of shared domain changes
    :return: False if the problem is not consistent
    """
    statistics[STATS_PROBLEM_FILTERS_NB] += 1
    init_propagators_to_filter(
        propagators_to_filter, changes, propagator_nb, propagator_variable_bounds, propagator_indices
    )
    while True:
        none = True
        for prop_idx in range(propagator_nb):
            if propagators_to_filter[prop_idx]:
                propagators_to_filter[prop_idx] = none = False
                break
        if none:
            return True
        statistics[STATS_PROBLEM_PROPAGATORS_FILTERS_NB] += 1
        prop_variable_bounds = propagator_variable_bounds[prop_idx]
        prop_data_bounds = propagator_data_bounds[prop_idx]
        prop_indices = propagator_indices[prop_variable_bounds[START] : prop_variable_bounds[END]]
        prop_offsets = propagator_offsets[prop_variable_bounds[START] : prop_variable_bounds[END]].reshape((-1, 1))
        prop_domains = shared_domains[prop_indices]
        np.add(prop_domains, prop_offsets, prop_domains)
        prop_data = propagator_data[prop_data_bounds[START] : prop_data_bounds[END]]
        prop_new_domains = compute_domains(propagator_algorithms[prop_idx], prop_domains, prop_data)
        if prop_new_domains is None:
            return False
        old_shared_domains = shared_domains.copy()
        np.subtract(prop_new_domains, prop_offsets, prop_new_domains)
        shared_domains[prop_indices] = prop_new_domains
        update_propagators_to_filter(
            propagators_to_filter,
            old_shared_domains != shared_domains,
            propagator_nb,
            propagator_variable_bounds,
            propagator_indices,
            prop_idx,
        )


@jit(nopython=True, cache=True)
def init_propagators_to_filter(
    propagators_to_filter: NDArray,
    changes: Optional[NDArray],
    propagator_nb: int,
    propagator_bounds: NDArray,
    propagator_indices: NDArray,
) -> None:
    if changes is None:  # this is an initialization
        propagators_to_filter.fill(True)
    else:
        for prop_idx in range(propagator_nb):
            prop_bounds = propagator_bounds[prop_idx]
            propagators_to_filter[prop_idx] = np.any(changes[propagator_indices[prop_bounds[START] : prop_bounds[END]]])


@jit(nopython=True, cache=True)
def update_propagators_to_filter(
    propagators_to_filter: NDArray,
    changes: NDArray,
    propagator_nb: int,
    propagator_bounds: NDArray,
    propagator_indices: NDArray,
    propagator_idx: int,
) -> None:
    for prop_idx in range(propagator_nb):
        if prop_idx != propagator_idx:
            prop_bounds = propagator_bounds[prop_idx]
            if np.any(changes[propagator_indices[prop_bounds[START] : prop_bounds[END]]]):
                propagators_to_filter[prop_idx] = True


@jit(nopython=True, cache=True)
def compute_domains(algorithm: int, domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    Computes the new domains for the variables.
    :param domains: the initial domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    if algorithm == ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ:
        return alldifferent_lopez_ortiz_propagator.compute_domains(domains, data)
    elif algorithm == ALGORITHM_CONSTANT_SUM:
        return constant_sum_propagator.compute_domains(domains, data)
    elif algorithm == ALGORITHM_DUMMY:
        return dummy_propagator.compute_domains(domains, data)
    elif algorithm == ALGORITHM_MUL:
        return mul_propagator.compute_domains(domains, data)
    elif algorithm == ALGORITHM_SUM:
        return sum_propagator.compute_domains(domains, data)
    return None


@jit(nopython=True, cache=True)
def is_solved(shared_domains: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :return: a boolean
    """
    return bool(np.all(np.equal(shared_domains[:, MIN], shared_domains[:, MAX])))
